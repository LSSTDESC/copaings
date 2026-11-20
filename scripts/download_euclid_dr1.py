#!/usr/bin/env python
"""Download the Euclid DR1 MER catalog around ECDFS, one FITS file per MER tile.

Each tile is written as ``euclid_dr1_mer_final_{tile}.fits`` (columns upper case,
table in HDU 1), the same layout as the Q1 files ``euclid_q1_mer_final_{tile}.fits``,
so the existing reader only needs the new file prefix.

Source: ESA TAP (``https://eas.esac.esa.int/tap-server/tap``), table
``dr1.mer_catalogue_deep_survey`` (EDF-Fornax holds ECDFS). One sync query per
MER tile (``tile_index``), several tiles at once, resumable: tiles already on
disk are skipped. The tile list and the per-tile row counts are computed with the
sky cut, so every tile is checked for a complete download.

Default cut: 1 deg around the Rubin DP1 ECDFS field (RA 53.13, Dec -28.10), which
also contains the classic ECDFS (RA 53.12, Dec -27.81). Change it with --ra,
--dec, --radius to match the LSST footprint.

Examples::

    python scripts/download_euclid_dr1.py --validate-only     # check columns, exit
    python scripts/download_euclid_dr1.py --list-tiles        # tiles in the cut, exit
    python scripts/download_euclid_dr1.py --output /sdf/data/rubin/user/$USER/euclid_dr1_ecdfs

Requirements: numpy, pandas, astropy, pyvo, requests (pip install -e ".[download]").
"""

from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

TAP_URL = "https://eas.esac.esa.int/tap-server/tap"
DEFAULT_TABLE = "dr1.mer_catalogue_deep_survey"
FILE_PREFIX = "euclid_dr1_mer_final"  # Q1 files: euclid_q1_mer_final
TILE_COLUMN = "TILE_INDEX"

# ECDFS: Rubin DP1 field centre, deg
DEFAULT_RA, DEFAULT_DEC, DEFAULT_RADIUS = 53.13, -28.10, 1.0

# columns read by the analysis, plus OBJECT_ID and TILE_INDEX for bookkeeping
COLUMNS = [
    "OBJECT_ID",
    TILE_COLUMN,
    "RIGHT_ASCENSION",
    "DECLINATION",
    "POSITION_ANGLE",
    "SEMIMAJOR_AXIS",
    "ELLIPTICITY",
    "FLUX_VIS_1FWHM_APER",
    "FLUXERR_VIS_1FWHM_APER",
    "EXTENDED_FLAG",
]
FLOAT64_COLUMNS = ("RIGHT_ASCENSION", "DECLINATION")  # other floats -> float32
_INT_DTYPES = {"long": "Int64", "int": "Int32", "short": "Int16"}


# ------------------------------------------------------------- schema ----


def table_schema(service, table):
    """Column names and TAP datatypes of ``table``.

    Parameters
    ----------
    service : pyvo.dal.TAPService
        TAP service.
    table : str
        Table name, e.g. ``dr1.mer_catalogue_deep_survey``.

    Returns
    -------
    dict
        ``{column_name: datatype}``, names as the service gives them.

    Raises
    ------
    SystemExit
        Table not in ``TAP_SCHEMA``.
    """
    # TAP_SCHEMA, not "SELECT TOP 1 *": ESA DR1 VOTable breaks astropy (E01)
    schema = service.search(
        "SELECT column_name, datatype FROM TAP_SCHEMA.columns "
        f"WHERE table_name = '{table}'"
    ).to_table()
    if len(schema) == 0:
        raise SystemExit(f"{table} not found in TAP_SCHEMA")
    return dict(zip(schema["column_name"], schema["datatype"]))


def match_columns(available, wanted):
    """Map wanted upper-case names to the source names, ignoring case.

    Parameters
    ----------
    available : iterable of str
        Source column names.
    wanted : list of str
        Wanted names, upper case.

    Returns
    -------
    dict
        ``{source_name: wanted_name}``.

    Raises
    ------
    ValueError
        Some wanted columns missing.
    """
    lookup = {name.upper(): name for name in available}
    missing = [c for c in wanted if c not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} column(s) not in the source: {missing}")
    return {lookup[c]: c for c in wanted}


def column_dtypes(datatypes, mapping):
    """pandas dtypes of the selected columns, from their TAP datatypes.

    Integers nullable (``Int64``...), floats float32 except RA/Dec.

    Parameters
    ----------
    datatypes : dict
        ``{source_name: TAP datatype}``.
    mapping : dict
        ``{source_name: wanted_name}``.

    Returns
    -------
    dict
        ``{source_name: pandas dtype}``.
    """
    dtypes = {}
    for source, name in mapping.items():
        datatype = datatypes[source]
        if datatype in _INT_DTYPES:
            dtypes[source] = _INT_DTYPES[datatype]
        elif datatype in ("double", "float"):
            dtypes[source] = "float64" if name in FLOAT64_COLUMNS else "float32"
        elif datatype == "boolean":
            dtypes[source] = "boolean"
        else:
            raise ValueError(f"{source}: TAP datatype {datatype!r} not handled")
    return dtypes


# ----------------------------------------------------------------- cut ----


def cone_cut(ra_col, dec_col, ra, dec, radius):
    """ADQL cone condition.

    Parameters
    ----------
    ra_col, dec_col : str
        Source names of the RA and Dec columns.
    ra, dec, radius : float
        Cone centre and radius, deg.

    Returns
    -------
    str
        ADQL condition.
    """
    return (
        f"1 = CONTAINS(POINT('ICRS', {ra_col}, {dec_col}), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius}))"
    )


# --------------------------------------------------------------- tiles ----


def tile_list(url, table, tile_col, where, path):
    """Tiles with sources in the cut, and their row counts in the cut.

    One sync query (small result; ESA dropped the async result download).
    Cached in ``path`` (CSV) together with the cut; a different cut recomputes.

    Parameters
    ----------
    url : str
        TAP service URL.
    table, tile_col, where : str
        Table, source name of the tile column, ADQL cut.
    path : Path
        Cache file.

    Returns
    -------
    pandas.DataFrame
        Columns ``tile``, ``n``.
    """
    stamp = path.with_suffix(".where")
    if path.exists() and stamp.exists() and stamp.read_text() == where:
        return pd.read_csv(path)
    content = run_sync(
        url,
        f"SELECT {tile_col} AS tile, COUNT(*) AS n FROM {table} "
        f"WHERE {where} GROUP BY {tile_col}",
    )
    tiles = pd.read_csv(io.BytesIO(content))
    tiles.columns = [c.lower() for c in tiles.columns]  # tile, n
    if tiles.empty:
        raise SystemExit("no source in the cut: check --ra, --dec, --radius")
    tiles = tiles.sort_values("tile").reset_index(drop=True)
    tiles.to_csv(path, index=False)
    stamp.write_text(where)
    return tiles


# ------------------------------------------------------------ download ----


def run_sync(url, query, retries=3):
    """Run one sync TAP query, CSV result.

    Retries after 30 s, 60 s... POST, so long column lists fit.

    Parameters
    ----------
    url : str
        TAP service URL.
    query : str
        ADQL query.
    retries : int
        Attempts.

    Returns
    -------
    bytes
        CSV content.
    """
    import requests

    params = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(f"{url}/sync", data=params, timeout=1800)
            response.raise_for_status()
            if response.content.startswith(b"<"):  # VOTable error, not CSV
                raise RuntimeError(response.text[:1000])
            return response.content
        except (requests.RequestException, RuntimeError):
            if attempt == retries:
                raise
            time.sleep(30 * attempt)


def write_fits(frame, path):
    """Write a DataFrame as a FITS binary table (HDU 1), atomically.

    Nullable columns without nulls become plain numpy columns; columns with
    nulls stay masked (FITS TNULL).

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to write.
    path : Path
        Output file.
    """
    from astropy.table import Table

    for name in frame.columns:
        col = frame[name]
        if pd.api.types.is_extension_array_dtype(col) and not col.isna().any():
            frame[name] = col.to_numpy(dtype=col.dtype.numpy_dtype)  # drop mask
    tmp = path.with_name(path.name + ".tmp")
    Table.from_pandas(frame).write(tmp, format="fits", overwrite=True)
    tmp.rename(path)  # file exists only when complete


def download_tile(url, query, dtypes, mapping, path, expected):
    """Download one tile and write it as FITS.

    Parameters
    ----------
    url, query : str
        TAP URL and ADQL query of the tile.
    dtypes, mapping : dict
        From :func:`column_dtypes` and :func:`match_columns`.
    path : Path
        Output FITS file.
    expected : int
        Rows the tile must have in the cut.

    Returns
    -------
    int
        Rows written.

    Raises
    ------
    RuntimeError
        Row count differs from ``expected`` (truncated result).
    """
    content = run_sync(url, query)
    frame = pd.read_csv(io.BytesIO(content), dtype=dtypes).rename(columns=mapping)
    if len(frame) != expected:
        raise RuntimeError(f"{len(frame):,} rows instead of {expected:,}")
    write_fits(frame, path)
    return len(frame)


def download_all(url, table, mapping, dtypes, where, tiles, out_dir, parallel):
    """Download every tile not yet on disk, several at once.

    Parameters
    ----------
    url, table, where : str
        TAP URL, table, ADQL cut.
    mapping, dtypes : dict
        Column mapping and dtypes.
    tiles : pandas.DataFrame
        Columns ``tile``, ``n``.
    out_dir : Path
        Output folder.
    parallel : int
        Tiles at once.

    Returns
    -------
    list of int
        Failed tiles.
    """
    tile_col = next(s for s, w in mapping.items() if w == TILE_COLUMN)
    select = ", ".join(mapping)
    files = {t: out_dir / f"{FILE_PREFIX}_{t}.fits" for t in tiles["tile"]}
    todo = tiles[[not files[t].exists() for t in tiles["tile"]]]
    print(
        f"{len(tiles):,} tiles ({tiles['n'].sum():,} sources), "
        f"{len(tiles) - len(todo):,} already done, {parallel} at once"
    )

    t0, n_done, failed = time.time(), 0, []
    with ProcessPoolExecutor(parallel) as pool:
        futures = {
            pool.submit(
                download_tile,
                url,
                f"SELECT {select} FROM {table} WHERE {tile_col} = {t} AND {where}",
                dtypes,
                mapping,
                files[t],
                n,
            ): (t, n)
            for t, n in zip(todo["tile"], todo["n"])
        }
        for k, future in enumerate(as_completed(futures), 1):
            tile, n = futures[future]
            try:
                rows = future.result()
            except Exception as error:  # reported; next run retries
                failed.append(tile)
                print(f"[{k}/{len(todo)}] tile {tile} FAILED: {error}", flush=True)
                continue
            n_done += rows
            rate = n_done / (time.time() - t0)
            print(
                f"[{k}/{len(todo)}] tile {tile}: {rows:,} rows | {rate:,.0f}/s",
                flush=True,
            )
    return failed


# ---------------------------------------------------------------- main ----


def parse_args(argv=None):
    """Command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("euclid_dr1_ecdfs"),
        help="output folder of the tile FITS files",
    )
    ap.add_argument("--tap", default=TAP_URL, help="TAP service URL")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="DR1 MER table")
    ap.add_argument("--ra", type=float, default=DEFAULT_RA, help="cone RA, deg")
    ap.add_argument("--dec", type=float, default=DEFAULT_DEC, help="cone Dec, deg")
    ap.add_argument(
        "--radius", type=float, default=DEFAULT_RADIUS, help="cone radius, deg"
    )
    ap.add_argument(
        "--where",
        default=None,
        help="extra ADQL cut, source column names (ANDed with the cone)",
    )
    ap.add_argument(
        "--parallel-tiles",
        type=int,
        default=8,
        help="tiles at once (8 is near ESA's maximum speed)",
    )
    ap.add_argument(
        "--validate-only", action="store_true", help="check the columns, then exit"
    )
    ap.add_argument(
        "--list-tiles",
        action="store_true",
        help="print the tiles in the cut, then exit",
    )
    return ap.parse_args(argv)


def main(argv=None):
    """Validate columns, list tiles in the cut, download them."""
    import pyvo

    args = parse_args(argv)
    service = pyvo.dal.TAPService(args.tap)

    datatypes = table_schema(service, args.table)
    mapping = match_columns(datatypes, COLUMNS)
    dtypes = column_dtypes(datatypes, mapping)
    print(f"{args.table}: {len(mapping)} columns validated")
    if args.validate_only:
        return

    source = {w: s for s, w in mapping.items()}  # wanted -> source name
    where = cone_cut(
        source["RIGHT_ASCENSION"], source["DECLINATION"], args.ra, args.dec, args.radius
    )
    if args.where:
        where = f"{where} AND ({args.where})"

    args.output.mkdir(parents=True, exist_ok=True)
    tiles = tile_list(
        args.tap, args.table, source[TILE_COLUMN], where, args.output / "tiles.csv"
    )
    if args.list_tiles:
        print(tiles.to_string(index=False))
        print(f"{len(tiles)} tiles, {tiles['n'].sum():,} sources")
        return

    failed = download_all(
        args.tap,
        args.table,
        mapping,
        dtypes,
        where,
        tiles,
        args.output,
        args.parallel_tiles,
    )
    if failed:
        raise SystemExit(f"{len(failed)} tile(s) failed: {failed}. Run again.")
    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
