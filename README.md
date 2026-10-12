# copaings

Pipeline to measure **unrecognized blends** and the **blending entropy S_b** ([arXiv:2609.12158](https://arxiv.org/abs/2609.12158)) of LSST (DP1 / DP2) object catalogs against the higher-resolution
Euclid Q1 (MER) and HST (HLF GOODS-S) catalogs, built to run as parallel slurm jobs and to compare many catalog variants
(for example visit-subset reruns). It is a restructuring and extension of the original analysis notebook, using the
[`friendly`](https://github.com/LSSTDESC/friendly) package for the matching / ellipse / entropy code (see *Dependencies*).

What it computes, for each LSST object: whether it is part of a group where several true galaxies fall on fewer LSST detections
(unrecognized blend), and its blending entropy S_b, the entropy of the probabilities that it matches each overlapping true galaxy
(from ellipse overlaps and magnitude differences). The blending entropy is described, used and tested in
[arXiv:2609.12158](https://arxiv.org/abs/2609.12158); the entropy itself is computed by `friendly`.

Start with **`notebooks/copaings_demo.ipynb`**: it runs every stage in a cell on a small piece of sky and ends with the S_b curve.

## Pipeline

| stage | what it does | main output (`<out_root>/<name>/`) |
|---|---|---|
| `prep` | LSST (butler), Euclid and HST catalogs, cuts, footprints, covered areas | `catalogs/{lsst,euclid,hst}.parquet`, `meta.json` |
| `fof` | friends-of-friends matching, one job per survey | `groups/{E,H}_fof.pkl` |
| `eot` | ellipse overlap test: split each FoF group into groups of really overlapping ellipses (slurm array) | `groups/*_eot_part_*.pkl` |
| `eot_merge` | merge the parts | `groups/{E,H}_no_overlap.pkl` |
| `magfit` | magnitudes of all LSST objects in the Euclid VIS / HST F814 bands (tree on ugrizy + size) | `catalogs/lsst_fit.parquet`, `fit_params.json`, `*_tree.pkl` |
| `sb` | blending entropy for the objects of groups that are not 1-1 (slurm array) | `sb/*_part_*.parquet` |
| `sb_merge` | assemble S_b and purity for every LSST object | `lsst_sb.parquet`, `sb/sb_merge_stats.json` |
| `plots` | standard figures | `plots/*.pdf` |

Group indices are row positions in the `prep` parquet files; all later stages read those files, never the butler.

### S_b conventions (`lsst_sb.parquet`, columns `blending_entropy_{Euclid,HST}`)

| value | meaning |
|---|---|
| >= 0 | computed entropy (object in a group other than 1-1) |
| 0 | object in a 1-1 group after the overlap test: a single candidate, nothing ambiguous |
| -1 | object overlaps no galaxy of that survey (outside the survey, no neighbour, or no ellipse overlap) |
| NaN | only if something failed; a warning is printed and the count saved in `sb/sb_merge_stats.json` |

`purity_*` follows the same rule (1, -1) but can still be NaN for objects whose LSST second moments are undefined.

## Install

```bash
git clone https://github.com/LSSTDESC/copaings.git && cd copaings
pip install -e .            # numpy pandas scipy astropy healpy pyyaml matplotlib scikit-learn tqdm pyarrow networkx
pip install -e ".[test]"    # pytest
```

### Dependencies you must provide

* **[`friendly`](https://github.com/LSSTDESC/friendly)** (matching, ellipses, graphs, entropy). Either `pip install` it, or point `FRIENDLY_PATH` to the directory that contains the package.
* **LSST science pipelines + butler**, only for `prep` (the other stages run without them: source `loadLSST.bash`, no `setup`).
* Euclid Q1 MER tile catalogs (`euclid_dir`) and the HST HLF catalog (`hst_file`); the defaults are the SLAC S3DF paths.

## Use

```bash
python -m copaings prep      cases/demo_dp2_box.yaml
python -m copaings fof       cases/demo_dp2_box.yaml --tel Euclid
python -m copaings eot       cases/demo_dp2_box.yaml --tel Euclid --all     # or --task-id N inside a slurm array
python -m copaings eot_merge cases/demo_dp2_box.yaml --tel Euclid
...                                                                          # same for HST, then magfit, sb, sb_merge, plots

python -m copaings submit  cases/dp2_14tracts_v3.yaml                       # the whole chain on slurm (afterok dependencies)
python -m copaings submit  cases/dp2_14tracts_v3.yaml --from-stage magfit   # earlier stages assumed done
python -m copaings submit  cases/dp2_14tracts_v3.yaml --skip-stages fof,eot,eot_merge --dry-run
python -m copaings compare cases/a.yaml cases/b.yaml --out compare_ab       # overlay several cases
```

A *case* is a yaml file; unset fields take the defaults of `copaings/config.py`. The case `name` fixes the output directory, so give each variation its own name.

| field | default | meaning |
|---|---|---|
| `repo`, `collections`, `skymap`, `dataset_type`, `tracts` | DP1 | where the LSST catalog is read |
| `lsst_columns` | `{}` | overrides of the butler column names (see `DEFAULT_LSST_COLUMNS`) |
| `linking_length` | 2.0 | FoF radius in arcsec (3.0 recommended: 2" misses about 10% of the Euclid blends, 3" captures nearly all of them) |
| `mag_cut` | 35 | magnitude cut on all three catalogs |
| `euclid_region` | `auto` | all Euclid tiles over the LSST footprint; `notebook` = the 6 original tiles in a ~2.5 deg2 box |
| `only_hst_field`, `sky_box` | off | crop Euclid to the HST field / crop everything to `[ra0, ra1, dec0, dec1]` |
| `eot_dense_max` | 400 | FoF groups above this size use the KD-tree ellipse test; 0 = always |
| `fast_sb_mc` | true | fast, validated replacement of friendly's Monte Carlo overlap area |
| `n_eot_tasks`, `n_sb_tasks` | 100 | slurm array sizes |
| `setup_cmd`, `account`, `partition`, `mem`, `mem_fof`, `time_*`, `exclude_nodes` | | slurm settings; pin the LSST stack in `setup_cmd` |

Examples in `cases/`: `example.yaml` (generic template), `dp1_example.yaml`, `demo_dp2_box.yaml` (used by the notebook).

## Tests

```bash
pytest          # needs friendly; checks the fast EOT, the fast overlap area, the S_b conventions, id collisions
```

## Layout

```
copaings/       package: config, catalogs (prep), stages, eot_fast, mc_fast, sb_core, plots, submit, cli
cases/           example case files
notebooks/       copaings_demo.ipynb
LICENSE          MIT
tests/
```

## Known limitations

* Cases built from different visit selections cover different sky; compare them on a common footprint.
* The magnitude trees use LSST sizes, which neighbours inflate in blends (check residuals against blendedness before trusting S_b of strongly blended objects).
* DP2 `i_blendedness` is not on the DP1 0-1 scale; the blendedness plots assume DP1.
* The pipeline was developed on SLAC S3DF; paths and slurm options are site specific.

## Credits

Matching, ellipse and entropy machinery: the [`friendly`](https://github.com/LSSTDESC/friendly) package (not part of this repository; check its licence).

## References

* Blending entropy: [arXiv:2609.12158](https://arxiv.org/abs/2609.12158), where it is described, used and tested.
* `friendly`: https://github.com/LSSTDESC/friendly

## License

MIT, see `LICENSE`.
