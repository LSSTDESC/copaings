# copaings

copaings measures **unrecognized blends** and the **blending entropy S_b** ([arXiv:2609.12158](https://arxiv.org/abs/2609.12158))
of LSST object catalogs (DP1, DP2). It does this by matching them to higher-resolution "true" catalogs from Euclid (Q1 or DR1 MER) and HST.
It runs as a chain of slurm jobs and can compare many catalog variants, for example visit-subset reruns. The matching, ellipse and entropy
code comes from the [`friendly`](https://github.com/LSSTDESC/friendly) package.

For each LSST object, copaings computes:

* whether it is an **unrecognized blend**: it belongs to a group where several true galaxies fall on fewer LSST detections;
* its **blending entropy S_b**: the entropy of the probabilities that it matches each overlapping true galaxy. These probabilities
  come from ellipse overlaps and magnitude differences. S_b is described, used and tested in
  [arXiv:2609.12158](https://arxiv.org/abs/2609.12158); `friendly` computes it.

New here? Start with **`notebooks/copaings_demo.ipynb`**. It runs every stage, one cell per stage, on a small piece of sky (a few minutes)
and ends with the S_b curve.

## Quick start (USDF / SLAC S3DF)

```bash
git clone https://github.com/LSSTDESC/copaings.git
git clone https://github.com/LSSTDESC/friendly.git        # next to copaings: copaings finds it there
cd copaings

# edit the TODO lines of the case (at least `account`), then submit the whole chain:
python -m copaings submit cases/lsst_dp2_euclid_dr1_hst_ecdfs.yaml
```

The results end up in `<out_root>/<name>/`. The figures are in `<out_root>/<name>/plots/`.

> Run every command from the root of the copaings clone. The slurm jobs `cd` there, so a relative `out_root`
> (like the default `./copaings_runs`) is resolved from there.

## Ready-to-run cases

A **case** is a yaml file that describes one run: which catalogs to read, how to match them and how to run on slurm.
The `cases/` folder has one case per field and data release, ready for USDF:

| case file | LSST | true catalogs | sky area |
|---|---|---|---|
| `lsst_dp2_euclid_dr1_hst_ecdfs.yaml` | DP2 (LSSTCam), 16 tracts | Euclid DR1 MER + HST HLF GOODS-S | ECDFS, circle of 1.6 deg around (53.0, -28.15) |
| `lsst_dp1_euclid_q1_hst_ecdfs.yaml` | DP1 (ComCam) | Euclid Q1 MER + HST HLF GOODS-S | ECDFS, all of DP1 |
| `lsst_dp2_hst_cosmos.yaml` | DP2 (LSSTCam), 9 tracts | HST ACS F814W (Leauthaud+07) | COSMOS, box of the ACS field |
| `demo_dp2_box.yaml` | DP2, 1 tract | Euclid Q1 + HST | ~0.07 deg2 box, used by the demo notebook |

In each file, the lines marked `TODO` are the ones you will most likely need to change:

* `account`: your slurm account. List yours with `sacctmgr show assoc user=$USER format=account%30`.
* `out_root`: where the outputs go. Full runs are large, so prefer group or scratch space over your home.
* outside USDF: the data paths (`repo`, `euclid_dir`, `hst_file`) and `setup_cmd`.

For a variation, copy a case under a new file name **and change `name`**. `name` sets the output directory, so two cases
with the same name write over each other's results. To keep your own cases out of git, add them to `.gitignore`.

## Install

**At USDF nothing needs to be installed.** The LSST stack has every dependency, and copaings and friendly are used from their clones
(see *Quick start*).

**Elsewhere**, or to get `copaings` as a command:

```bash
pip install -e .                  # numpy pandas scipy astropy healpy pyyaml matplotlib scikit-learn tqdm pyarrow networkx
pip install -e ".[test]"          # + pytest
pip install -e ".[download]"      # + pyvo, requests, for scripts/download_euclid_dr1.py
```

### What you must provide

* **[`friendly`](https://github.com/LSSTDESC/friendly)** (matching, ellipses, graphs, entropy). copaings looks for it, in this order, in
  the folder given by `FRIENDLY_PATH`, in the installed packages, and next to the copaings clone (`<folder of copaings>/friendly`),
  see `copaings/deps.py`. Its only dependencies are numpy, scipy, pandas, networkx and tqdm.
* **The LSST science pipelines and a butler**, only for the `prep` stage. The other stages only source `loadLSST.bash`, without `setup`.
  At USDF, `prep` sets the butler repository index and the database credentials itself when they are missing.
* **The true catalogs** (see below).

### Data

| catalog | case fields | at USDF | elsewhere |
|---|---|---|---|
| Euclid Q1 MER | `euclid_dir` | `/sdf/group/rubin/shared/euclid/q1/catalogs` (default) | a folder of `euclid_q1_mer_final_<tile>.fits` files |
| Euclid DR1 MER | `euclid_dir` | `/sdf/group/rubin/user/mesquita/euclid_dr1_ecdfs` (readable by `rubin_users`) | `python scripts/download_euclid_dr1.py --output <folder>` (ESA TAP, resumable) |
| HST HLF GOODS-S v2.1 | `hst_file`, `hst_catalog: hlf_goodss` | `/sdf/data/rubin/shared/hst/ecdfs/hlsp_hlf_hst_60mas_goodss_v2.1_catalog.fits` (default) | [MAST HLSP HLF](https://archive.stsci.edu/hlsp/hlf) |
| HST COSMOS ACS F814W | `hst_file`, `hst_catalog: cosmos_acs` | `/sdf/data/rubin/shared/hst/cosmos/catalogs/cosmos_acs_iphot_200709.parq` | Leauthaud et al. 2007 catalog (IRSA), as parquet with lower-case columns |

Each Euclid folder must hold a single release; the release is read from the file names. The tile files are FITS tables (HDU 1) with
the upper-case columns listed in `COLUMNS` of `scripts/download_euclid_dr1.py`. By default, the download script takes 1 deg around
the DP1 ECDFS field (RA 53.13, Dec -28.10); change it with `--ra`, `--dec`, `--radius`.

## Pipeline

| stage | what it does | main output (`<out_root>/<name>/`) |
|---|---|---|
| `prep` | reads the LSST (butler), Euclid and HST catalogs; applies the cuts; computes footprints and covered areas | `catalogs/{lsst,euclid,hst}.parquet`, `meta.json` |
| `fof` | friends-of-friends matching, one job per true catalog | `groups/{E,H}_fof.pkl` |
| `eot` | ellipse overlap test: splits each FoF group into groups of really overlapping ellipses (slurm array) | `groups/*_eot_part_*.pkl` |
| `eot_merge` | merges the parts | `groups/{E,H}_no_overlap.pkl` |
| `magfit` | magnitudes of all LSST objects in the Euclid VIS / HST F814W bands (tree on ugrizy + size) | `catalogs/lsst_fit.parquet`, `fit_params.json`, `*_tree.pkl` |
| `sb` | blending entropy of the objects in groups that are not 1-1 (slurm array) | `sb/*_part_*.parquet` |
| `sb_merge` | assembles S_b and purity for every LSST object | `lsst_sb.parquet`, `sb/sb_merge_stats.json` |
| `plots` | standard figures (sky map, group counts, unrecognized fraction, S_b) | `plots/{pdf,png}/*` |

`E` and `H` in the file names stand for Euclid and HST. Group indices are row positions in the `prep` parquet files. All later stages
read those files, never the butler.

### S_b conventions (`lsst_sb.parquet`)

For each true survey `<tel>` (`Euclid`, `HST`), `blending_entropy_<tel>` is a number only where the entropy is defined and
NaN elsewhere. The column `sb_flag_<tel>` says which case each object is in:

| `sb_flag_<tel>` | S_b | meaning |
|---|---|---|
| `computed` | >= 0 | entropy computed by `sb` (object in a group other than 1-1) |
| `one_to_one` | 0 | 1-1 group after the overlap test: a single candidate, nothing ambiguous (purity 1) |
| `out_footprint` | NaN | outside the footprint of the survey (Euclid: convex hull; HST: no HST galaxy within `hst_footprint_radius`) |
| `no_fof_match` | NaN | in the footprint, but no galaxy of the survey within the FoF linking length |
| `no_ellipse_overlap` | NaN | in a FoF group, but its ellipse overlaps no galaxy |
| `obj_only_neighbours` | NaN | in a group, but overlaps only other LSST objects, no galaxy |
| `zero_overlap` | NaN | in a group, touches galaxies but every overlap area is 0 |
| `failed` | NaN | the computation failed (e.g. missing magnitude); a warning is printed |

`purity_<tel>` is NaN for the objects in no group (`out_footprint`, `no_fof_match`, `no_ellipse_overlap`).
The number of objects with each flag is saved in `sb/sb_merge_stats.json`.

## Running

```bash
# the whole chain on slurm, each job waiting for the previous ones (afterok)
python -m copaings submit  cases/lsst_dp2_hst_cosmos.yaml
python -m copaings submit  cases/lsst_dp2_hst_cosmos.yaml --dry-run                         # only write the sbatch scripts
python -m copaings submit  cases/lsst_dp2_hst_cosmos.yaml --from-stage magfit               # earlier stages already done
python -m copaings submit  cases/lsst_dp2_hst_cosmos.yaml --skip-stages fof,eot,eot_merge

# one stage at a time, in the current process (small runs, debugging)
python -m copaings prep      cases/demo_dp2_box.yaml
python -m copaings fof       cases/demo_dp2_box.yaml --tel Euclid
python -m copaings eot       cases/demo_dp2_box.yaml --tel Euclid --all     # or --task-id N inside a slurm array
python -m copaings eot_merge cases/demo_dp2_box.yaml --tel Euclid
...                                                                          # same for HST, then magfit, sb, sb_merge, plots

# other commands
python -m copaings compare   cases/a.yaml cases/b.yaml --out compare_ab     # overlay the results of several cases
python -m copaings footprint cases/a.yaml                                   # LSST positions before the sky cuts
```

`submit` writes a frozen copy of the case to `<out_root>/<name>/case.yaml`, plus the sbatch scripts and logs to `slurm/`.
If a job fails, the jobs that depend on it never start: `squeue` shows them as `DependencyNeverSatisfied`. Cancel them
(`scancel`), fix the problem and resubmit with `--from-stage`.
After `pip install`, `copaings <command> ...` works the same as `python -m copaings <command> ...`.

## Case fields

Only `name` is required; unset fields take the defaults of `copaings/config.py`, which documents every field.

| field | default | meaning |
|---|---|---|
| `name` | required | case name; outputs go to `<out_root>/<name>/` |
| `out_root` | `./copaings_runs` | parent folder of all case outputs |
| `repo`, `collections`, `skymap`, `dataset_type`, `tracts` | DP1 | where the LSST catalog is read; tracts without data are skipped |
| `lsst_columns` | `{}` | overrides of the butler column names (see `DEFAULT_LSST_COLUMNS`) |
| `common_frame_xy` | `true` | rebuild LSST `x`, `y` in one tangent plane (per-tract pixel frames do not agree across tracts) |
| `telescopes` | `[Euclid, HST]` | true catalogs to use; `[HST]` for a field without Euclid (COSMOS) |
| `euclid_dir` | Q1 at S3DF | folder of the Euclid tile files `euclid_<release>_mer_final_<tile>.fits` (Q1 or DR1, one release per folder) |
| `euclid_region` | `auto` | Q1: the Euclid tiles over the LSST footprint; DR1 (tiles not named by healpix): every file of `euclid_dir`; `all` = every file; `notebook` = the 6 original Q1 tiles in a ~2.5 deg2 box |
| `hst_file`, `hst_catalog` | HLF GOODS-S, `hlf_goodss` | HST catalog and its format: `hlf_goodss` (ECDFS) or `cosmos_acs` (COSMOS ACS F814W, Leauthaud+07, `mu_class == 1`) |
| `hst_footprint_radius` | 5.0 | an LSST object is in the HST footprint if an HST galaxy lies within this many arcsec; 0 = convex hull. The HST fields are jagged, so their hull includes sky with no HST data. HLF rows without F814W coverage (`f_f814w = -99`, `w_f814w = 0`) are always dropped |
| `sky_circle` | off | cut all catalogs to the circle `[ra_center, dec_center, radius]` (deg), so every survey covers the same area |
| `sky_box` | off | cut all catalogs to `[ra_min, ra_max, dec_min, dec_max]` (deg) |
| `only_hst_field` | `false` | cut Euclid to the HST field |
| `linking_length` | 2.0 | FoF radius in arcsec. 3.0 is recommended (all provided cases use it): 2" misses about 10% of the Euclid blends, 3" captures nearly all of them |
| `mag_cut` | 35 | magnitude cut on all catalogs |
| `eot_dense_max` | 400 | FoF groups above this size use the KD-tree ellipse test; 0 = always |
| `n_eot_tasks`, `n_sb_tasks` | 100 | slurm array sizes |
| `setup_cmd` | | shell command that sets up the LSST stack (used as is by `prep`; the other stages use the part before the first `&&`) |
| `account`, `partition`, `mem`, `mem_fof`, `time_short`, `time_long`, `exclude_nodes` | | slurm settings; `mem_fof` defaults to `mem` |

## Tests

```bash
pytest          # needs friendly; checks the fast EOT, friendly's fast functions (overlap area, FoF), the S_b conventions, id collisions
```

## Layout

```
copaings/       the package: config (case fields), catalogs (prep), stages, eot_fast, sb_core, plots, style, submit, cli,
                deps (finds friendly)
cases/          case files (see "Ready-to-run cases")
scripts/        download_euclid_dr1.py: Euclid DR1 MER tiles from the ESA archive
notebooks/      copaings_demo.ipynb
tests/
LICENSE         MIT
```

## Known limitations

* Cases built from different visit selections cover different sky; compare them on a common footprint.
* The magnitude trees use LSST sizes, which neighbours inflate in blends. Check the residuals against blendedness before trusting
  the S_b of strongly blended objects.
* DP2 `i_blendedness` is not on the DP1 0-1 scale; the blendedness plots assume DP1.
* The pipeline was developed on SLAC S3DF; the default paths and slurm options are specific to that site.

## Credits

The matching, ellipse and entropy machinery comes from the [`friendly`](https://github.com/LSSTDESC/friendly) package
(not part of this repository; check its licence).

## References

* Blending entropy: [arXiv:2609.12158](https://arxiv.org/abs/2609.12158), where it is described, used and tested.
* `friendly`: https://github.com/LSSTDESC/friendly
* HST HLF GOODS-S: Whitaker et al. 2019, ApJS 244, 16.
* HST COSMOS ACS: Leauthaud et al. 2007, ApJS 172, 219.

## License

MIT, see `LICENSE`.
