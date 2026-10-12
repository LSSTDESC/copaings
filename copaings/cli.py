"""Command line entry point.

  python -m copaings <stage> cases/foo.yaml [--tel Euclid] [--task-id N --n-tasks M | --all]
  python -m copaings submit cases/foo.yaml [--from-stage STAGE] [--dry-run]
  python -m copaings compare cases/a.yaml cases/b.yaml --out plots_compare

Array stages read SLURM_ARRAY_TASK_ID when --task-id is not given.
"""
import argparse
import os

from .config import Case

STAGES = ('prep', 'fof', 'eot', 'eot_merge', 'magfit', 'sb', 'sb_merge', 'plots')
ARRAY_STAGES = ('eot', 'sb')
TEL_STAGES = ('fof', 'eot', 'eot_merge', 'sb')


def run_stage(case, stage, tel=None, task_id=None, n_tasks=None, run_all=False):
    """Run one pipeline stage for a case.

    For the array stages (`eot`, `sb`) the slice is taken from `task_id`, from ``SLURM_ARRAY_TASK_ID``, or all slices
    are run one after the other with `run_all`.

    Parameters
    ----------
    case : Case
    stage : {'prep', 'fof', 'eot', 'eot_merge', 'magfit', 'sb', 'sb_merge', 'plots'}
        Stage to run.
    tel : {'Euclid', 'HST'}, optional
        Survey; needed by `fof`, `eot`, `eot_merge` and `sb`.
    task_id : int, optional
        Slice index of an array stage.
    n_tasks : int, optional
        Number of slices; default is `n_eot_tasks` or `n_sb_tasks` of the case.
    run_all : bool, default False
        Run all slices of an array stage serially (small tests, notebooks).

    Raises
    ------
    SystemExit
        If an array stage is run without `task_id`, `run_all` or a slurm array environment.
    """
    from . import stages
    if stage in ARRAY_STAGES:
        n_tasks = n_tasks or (case.n_eot_tasks if stage == 'eot' else case.n_sb_tasks)
        if run_all:  # interactive: run every slice one after the other
            for i in range(n_tasks):
                run_stage(case, stage, tel, task_id=i, n_tasks=n_tasks)
            return
        if task_id is None:
            if 'SLURM_ARRAY_TASK_ID' not in os.environ:
                raise SystemExit(f'{stage} is an array stage: pass --task-id N (0..{n_tasks - 1}) or --all, '
                                 'or run it through slurm (submit)')
            task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if stage == 'prep':
        stages.prep(case)
    elif stage == 'fof':
        stages.fof(case, tel)
    elif stage == 'eot':
        stages.eot(case, tel, task_id, n_tasks)
    elif stage == 'eot_merge':
        stages.eot_merge(case, tel, n_tasks or case.n_eot_tasks)
    elif stage == 'magfit':
        stages.magfit(case)
    elif stage == 'sb':
        stages.sb(case, tel, task_id, n_tasks)
    elif stage == 'sb_merge':
        stages.sb_merge(case, n_tasks or case.n_sb_tasks)
    elif stage == 'plots':
        from . import plots
        plots.all_plots(case)


def main(argv=None):
    """Command line entry point (``python -m copaings``).

    Parameters
    ----------
    argv : list of str, optional
        Arguments; default is ``sys.argv[1:]``. First the stage name (a pipeline stage, ``submit`` or ``compare``),
        then the case yaml file(s), then the options of that stage (see ``--help``).
    """
    ap = argparse.ArgumentParser(prog='copaings')
    ap.add_argument('stage', choices=STAGES + ('submit', 'compare'))
    ap.add_argument('cases', nargs='+', help='case yaml file(s)')
    ap.add_argument('--tel', choices=('Euclid', 'HST'))
    ap.add_argument('--task-id', type=int)
    ap.add_argument('--n-tasks', type=int)
    ap.add_argument('--all', action='store_true', help='array stages: run all slices serially (small tests)')
    ap.add_argument('--from-stage', choices=STAGES, default='prep',
                    help='submit: first stage to run (earlier stages must already be done)')
    ap.add_argument('--skip-stages', default='', help='submit: comma-separated stages to leave out, e.g. fof,eot,eot_merge')
    ap.add_argument('--after', default='', help='submit: comma-separated job ids the first jobs must also wait for')
    ap.add_argument('--dry-run', action='store_true', help='submit: only write the sbatch scripts')
    ap.add_argument('--out', default='plots_compare', help='compare: output dir')
    a = ap.parse_args(argv)

    if a.stage == 'compare':
        from . import plots
        plots.compare([Case.from_yaml(p) for p in a.cases], a.out)
        return
    case = Case.from_yaml(a.cases[0])
    if a.stage == 'submit':
        from . import submit
        submit.submit(case, dry_run=a.dry_run, from_stage=a.from_stage, skip=tuple(x for x in a.skip_stages.split(',') if x),
                      after=tuple(x for x in a.after.split(',') if x))
        return
    if a.stage in TEL_STAGES and not a.tel:
        ap.error(f'{a.stage} needs --tel')
    run_stage(case, a.stage, a.tel, a.task_id, a.n_tasks, run_all=a.all)


if __name__ == '__main__':
    main()
