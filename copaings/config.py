"""Per-case configuration and on-disk layout."""
import os
from dataclasses import dataclass, field, fields

import yaml

# DP1 'object' table column -> internal name. Override in the case yaml (`lsst_columns`)
# if your DP2 catalog uses different names.
DEFAULT_LSST_COLUMNS = {
    'ra': 'coord_ra',
    'dec': 'coord_dec',
    'x': 'i_centroid_x',
    'y': 'i_centroid_y',
    'Ixx_pixel_i': 'i_ixx',
    'Iyy_pixel_i': 'i_iyy',
    'Ixy_pixel_i': 'i_ixy',
    'psf_ixx': 'i_ixxPSF',
    'psf_iyy': 'i_iyyPSF',
    'psf_fwhm_i_from_butler': 'i_fwhm',
    'i_cModelFlux': 'i_cModelFlux',
    'i_cModelFluxErr': 'i_cModelFluxErr',
    'i_blendedness': 'i_blendedness',
    'i_extendedness': 'i_extendedness',
    'r_cModelFlux': 'r_cModelFlux',
    'r_cModelFluxErr': 'r_cModelFluxErr',
    'r_blendedness': 'r_blendedness',
    'u_cModelFlux': 'u_cModelFlux',
    'z_cModelFlux': 'z_cModelFlux',
    'y_cModelFlux': 'y_cModelFlux',
    'g_cModelFlux': 'g_cModelFlux',
    'g_cModelFluxErr': 'g_cModelFluxErr',
    'g_blendedness': 'g_blendedness',
}

# Options that no longer exist, ignored so that old case files still load. fast_eot: the slow overlap test was
# removed; fast_sb_mc: the fast Monte Carlo overlap area is now part of friendly.
_REMOVED_KEYS = ('fast_eot', 'fast_sb_mc')

TELESCOPES = ('Euclid', 'HST')  # name of the "true" catalogs; initial is used in file names


@dataclass
class Case:
    """Configuration of one run (a "case"): which catalogs to read, how to match them and how to run on slurm.

    A case is normally written as a yaml file and loaded with `Case.from_yaml`. Every field has a default and only
    `name` is required. The case name fixes the output directory ``<out_root>/<name>/``, so give every variation you
    want to keep its own name.

    Parameters
    ----------
    name : str
        Case name; outputs go to ``out_root/name``.
    out_root : str, default './copaings_runs'
        Parent directory of all case outputs.
    repo, collections, skymap, dataset_type : str
        Butler repository, collection(s), skymap and dataset type of the LSST object table.
    tracts : list of int
        Tracts to read.
    lsst_columns : dict
        Overrides of the butler column names, ``{internal_name: butler_name}``. Defaults are in
        `DEFAULT_LSST_COLUMNS`.
    lsst_pixel_scale : float, default 0.2
        LSST pixel scale [arcsec/pixel].
    common_frame_xy : bool, default True
        Rebuild the LSST pixel positions ``x``, ``y`` in one tangent plane from ra/dec (per-tract pixel frames are
        not consistent across tracts). The butler values are kept as ``x_tract``, ``y_tract`` when they exist.
    euclid_dir, hst_file : str
        Folder of the Euclid MER tile catalogs, files ``euclid_<release>_mer_final_<tile>.fits`` (Q1 or DR1, one
        release per folder), and path of the HST catalog fits file.
    hst_catalog : {'hlf_goodss', 'cosmos_acs'}, default 'hlf_goodss'
        Format of `hst_file`: the HLF GOODS-S catalog (ECDFS, fits) or the COSMOS ACS F814W catalog of Leauthaud et
        al. 2007 (``cosmos_acs_iphot_200709.parq``).
    hst_footprint_radius : float, default 5.0
        An LSST object is in the HST footprint if an HST galaxy lies within this distance [arcsec]. The HST fields
        are jagged and their convex hull includes sky with no HST data; 0 uses the convex hull anyway.
    only_hst_field : bool, default False
        Crop the Euclid catalog to the bounding box of the HST catalog.
    sky_box : list of float, default []
        Optional ``[ra_min, ra_max, dec_min, dec_max]`` [deg] applied to all three catalogs (tests, demos).
    sky_circle : list of float, default []
        Optional ``[ra_center, dec_center, radius]`` [deg] applied to all three catalogs, so that every survey
        covers the same area (exact great-circle distance).
    euclid_region : {'auto', 'all', 'notebook'}, default 'auto'
        ``'auto'`` reads every Euclid tile that has a file and contains LSST objects when the files are named by
        healpix pixel (Q1), and every file of `euclid_dir` otherwise (DR1, named by TILE_INDEX); ``'all'`` reads every
        file of `euclid_dir`; ``'notebook'`` reads the six hand-picked ECDFS Q1 tiles cropped to a box of about
        2.5 deg2 (first version of the analysis).
    linking_length : float, default 2.0
        Friends-of-friends linking length [arcsec].
    mag_cut : float, default 35.0
        Objects fainter than this (own band) are removed from the three catalogs.
    telescopes : list of str
        Subset of ``['Euclid', 'HST']``; e.g. ``['HST']`` for a field without Euclid data (COSMOS).
        Names of the "true" catalogs to process.
    eot_dense_max : int, default 400
        FoF groups with more members than this use the KD-tree pair search in the fast overlap test (0 = always).
    n_eot_tasks, n_sb_tasks : int, default 100
        Sizes of the slurm arrays of the `eot` and `sb` stages.
    setup_cmd : str
        Shell command that sets up the full LSST stack (used by `prep` only).
    setup_cmd_light : str
        Shell command for the other stages; default is the part of `setup_cmd` before the first ``&&``.
    partition, account : str
        Slurm partition and account.
    exclude_nodes : list of str
        Nodes to exclude (``--exclude``).
    mem, mem_fof : str
        Memory per job; `mem_fof` is used for the FoF jobs and defaults to `mem`.
    time_short, time_long : str
        Wall-time limits for light stages and for `fof`, `eot`, `sb`.

    See Also
    --------
    Case.from_yaml : build a case from a yaml file.
    """

    name: str
    out_root: str = './copaings_runs'

    # --- LSST catalog (butler)
    repo: str = '/repo/dp1'
    collections: str = 'LSSTComCam/DP1'
    skymap: str = 'lsst_cells_v1'
    dataset_type: str = 'object'
    tracts: list = field(default_factory=lambda: [5063])
    lsst_columns: dict = field(default_factory=dict)  # partial overrides of DEFAULT_LSST_COLUMNS
    lsst_pixel_scale: float = 0.2  # arcsec / pix
    # Recompute x, y in one tangent plane from ra/dec (per-tract pixel frames are inconsistent across tracts).
    # The original per-tract values are kept as x_tract, y_tract.
    common_frame_xy: bool = True

    # --- external catalogs
    euclid_dir: str = '/sdf/group/rubin/shared/euclid/q1/catalogs'
    hst_file: str = '/sdf/data/rubin/shared/hst/ecdfs/hlsp_hlf_hst_60mas_goodss_v2.1_catalog.fits'
    # Format of hst_file: 'hlf_goodss' (ECDFS) or 'cosmos_acs' (COSMOS ACS F814W, Leauthaud+07)
    hst_catalog: str = 'hlf_goodss'
    # HST footprint: an LSST object is in it if an HST galaxy lies within this distance [arcsec]. HST fields are jagged,
    # so their convex hull includes sky with no HST data. 0 = convex hull, like Euclid.
    hst_footprint_radius: float = 5.0
    only_hst_field: bool = False
    # Optional [ra_min, ra_max, dec_min, dec_max] (deg): keep only this part of the sky in all three catalogs.
    # Handy for tests and demos (a whole pipeline run in minutes). Empty = no cut.
    sky_box: list = field(default_factory=list)
    # Optional [ra_center, dec_center, radius] (deg): keep only the objects within this circle in all three catalogs,
    # so that LSST, Euclid and HST cover one homogeneous area. Empty = no cut.
    sky_circle: list = field(default_factory=list)
    # Which Euclid tiles to read: 'auto' = all tiles with a file that contain LSST objects (no crop) for Q1
    # healpix-named files, every file of euclid_dir otherwise (DR1); 'all' = every file of euclid_dir;
    # 'notebook' = the six hand-picked ECDFS tiles cropped to a ~2.5 deg2 box (first version of this code).
    euclid_region: str = 'auto'

    # --- selection / matching
    linking_length: float = 2.0  # arcsec
    mag_cut: float = 35.0
    telescopes: list = field(default_factory=lambda: list(TELESCOPES))

    # Groups up to this many members test all pairs in the fast EOT (an object with undefined moments then counts as
    # overlapping with the whole group, like friendly); bigger groups use a KD-tree where it only links to neighbours
    # within a few arcsec. 0 = always the KD-tree path (consistent across linking lengths, used by the FoF-length tests).
    eot_dense_max: int = 400

    # --- parallelism
    n_eot_tasks: int = 100
    n_sb_tasks: int = 100

    # --- slurm (used by `submit`)
    setup_cmd: str = ''  # full LSST stack, used by `prep` (butler): 'source .../loadLSST.bash && setup lsst_distrib'
    # Every other stage only needs numpy/pandas/scipy/friendly: skip the slow `setup lsst_distrib` (many `eups list`
    # calls on a shared filesystem). Default: the part of setup_cmd before the first '&&'.
    setup_cmd_light: str = ''
    partition: str = 'milano'
    account: str = ''
    exclude_nodes: list = field(default_factory=list)  # e.g. ['sdfmilan256'] to avoid a bad node
    mem: str = '32G'
    mem_fof: str = ''  # FoF holds LSST + Euclid catalogs and a KD-tree in one process; default = mem
    time_short: str = '01:00:00'
    time_long: str = '12:00:00'

    def __post_init__(self):
        """Check the options that later stages rely on."""
        unknown = [t for t in self.telescopes if t not in TELESCOPES]
        if unknown or not self.telescopes:
            raise ValueError(f'telescopes must be a non-empty subset of {list(TELESCOPES)}, got {self.telescopes}')
        self.telescopes = [t for t in TELESCOPES if t in self.telescopes]  # fixed order: Euclid, HST
        if self.hst_catalog not in ('hlf_goodss', 'cosmos_acs'):
            raise ValueError(f"hst_catalog must be 'hlf_goodss' or 'cosmos_acs', got {self.hst_catalog!r}")

    @classmethod
    def from_yaml(cls, path):
        """Build a case from a yaml file.

        Parameters
        ----------
        path : str
            Path of the yaml file. Keys are the field names of `Case`.

        Returns
        -------
        Case

        Raises
        ------
        ValueError
            If the file contains a key that is not a field of `Case`.
        """
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        for key in _REMOVED_KEYS:  # options that no longer exist; ignored so that old case.yaml files still load
            d.pop(key, None)
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f'unknown keys in {path}: {sorted(unknown)}')
        return cls(**d)

    # ---- derived
    @property
    def fof_mem(self):
        """Memory request of the FoF jobs.

        Returns
        -------
        str
            `mem_fof` if set, else `mem`.
        """
        return self.mem_fof or self.mem

    @property
    def light_setup(self):
        """Shell setup command of the stages that do not need the full LSST stack.

        Returns
        -------
        str
            `setup_cmd_light` if set, else the part of `setup_cmd` before the first ``&&`` (typically only sourcing
            ``loadLSST.bash``, skipping the slow ``setup lsst_distrib``).
        """
        return self.setup_cmd_light or self.setup_cmd.split('&&')[0].strip()

    @property
    def columns(self):
        """Butler column names used for this case.

        Returns
        -------
        dict
            ``{internal_name: butler_name}``: `DEFAULT_LSST_COLUMNS` updated with `lsst_columns`.
        """
        columns = dict(DEFAULT_LSST_COLUMNS)
        columns.update(self.lsst_columns)
        return columns

    @property
    def out_dir(self):
        """Output directory of the case.

        Returns
        -------
        str
            ``out_root/name``.
        """
        return os.path.join(self.out_root, self.name)

    def path(self, *parts, makedirs=True):
        """Path of a file inside the output directory of the case.

        Parameters
        ----------
        *parts : str
            Path components below `out_dir`, e.g. ``('groups', 'E_fof.pkl')``.
        makedirs : bool, default True
            Create the parent directory of the returned path if it does not exist.

        Returns
        -------
        str
        """
        p = os.path.join(self.out_dir, *parts)
        if makedirs:
            os.makedirs(os.path.dirname(p), exist_ok=True)
        return p


def initial(tel_name):
    """First letter of a survey name, used in file names.

    Parameters
    ----------
    tel_name : str
        ``'Euclid'`` or ``'HST'``.

    Returns
    -------
    str
        ``'E'`` or ``'H'``.
    """
    return tel_name[0]  # 'E' / 'H', same convention as the notebook


# Column of the LSST catalog holding the LSST magnitude expressed in each true survey's band
TRUE_MAG_COL_IN_LSST = {'Euclid': 'vis_mag', 'HST': 'F814_MAG'}
# Column of the true catalog holding its own magnitude
TRUE_MAG_COL = {'Euclid': 'vis_mag', 'HST': 'F814_MAG'}
TRUE_MAGERR_COL = {'Euclid': 'vis_magErr', 'HST': 'F814_MAGERR'}
