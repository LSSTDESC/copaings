"""Write and submit the slurm job chain for one case.

prep -> fof (per tel) -> eot array (per tel) -> eot_merge (per tel) -> magfit -> sb array (per tel) -> sb_merge -> plots
Each link uses --dependency=afterok, so a failed stage stops the chain; rerun `submit` after fixing (finished
stages are not skipped, run individual stages by hand with `python -m copaings <stage> ...` if you only need part).
"""
import dataclasses
import os
import subprocess

import yaml

from .config import Case, initial

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(case, name, command, time, array=None, cpus=1, mem=None, full_stack=False):
    """Write the sbatch script of one job.

    Parameters
    ----------
    case : Case
        Slurm settings; the script is written to ``slurm/<name>.sbatch`` of the case, logs go to ``slurm/logs/``.
    name : str
        Job name suffix (the job is called ``<case.name>_<name>``) and script name.
    command : str
        Arguments of ``python -m copaings``.
    time : str
        Wall-time limit.
    array : str, optional
        Slurm array specification, e.g. ``'0-99'``.
    cpus : int, default 1
        CPUs per task.
    mem : str, optional
        Memory; default `case.mem`.
    full_stack : bool, default False
        Use `case.setup_cmd` (full LSST stack) instead of `case.light_setup`.

    Returns
    -------
    str
        Path of the script.
    """
    log_dir = os.path.dirname(case.path('slurm', 'logs', 'x'))
    lines = ['#!/bin/bash',
             f'#SBATCH --job-name={case.name}_{name}',
             f'#SBATCH --partition={case.partition}',
             f'#SBATCH --time={time}',
             f'#SBATCH --cpus-per-task={cpus}',
             f'#SBATCH --mem={mem or case.mem}',
             f'#SBATCH --output={log_dir}/{name}_%A_%a.out' if array else f'#SBATCH --output={log_dir}/{name}_%j.out']
    if case.account:
        lines.append(f'#SBATCH --account={case.account}')
    if case.exclude_nodes:
        lines.append(f'#SBATCH --exclude={",".join(case.exclude_nodes)}')
    if array:
        lines.append(f'#SBATCH --array={array}')
    # no `set -u`: loadLSST.bash reads unset variables
    lines += ['', 'set -eo pipefail', (case.setup_cmd if full_stack else case.light_setup), f'cd {REPO_DIR}', f'python -m copaings {command}', '']
    path = case.path('slurm', f'{name}.sbatch')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path


class _Submitter:
    """Callable that submits sbatch scripts, or only prints the commands in dry-run mode.

    Parameters
    ----------
    dry_run : bool
        Do not call sbatch; return fake job ids ``DRY1``, ``DRY2``, ...
    """

    def __init__(self, dry_run):
        """Store the dry-run flag and reset the fake job counter."""
        self.dry_run = dry_run
        self.n = 0

    def __call__(self, script, after=()):
        """Submit one script.

        Parameters
        ----------
        script : str
            Path of the sbatch script.
        after : sequence of str, optional
            Job ids that must finish successfully first (``--dependency=afterok``).

        Returns
        -------
        str
            Job id (or a fake id in dry-run mode).
        """
        cmd = ['sbatch', '--parsable']
        if after:
            cmd.append('--dependency=afterok:' + ':'.join(after))
        cmd.append(script)
        if self.dry_run:
            self.n += 1
            print('DRY', ' '.join(cmd))
            return f'DRY{self.n}'
        job_id = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        print(f'{os.path.basename(script)} -> {job_id}')
        return job_id


STAGE_ORDER = ('prep', 'fof', 'eot', 'eot_merge', 'magfit', 'sb', 'sb_merge', 'plots')


def submit(case: Case, dry_run=False, from_stage='prep', skip=(), after=()):
    """Write and submit the slurm job chain of a case.

    The chain is ``prep`` -> ``fof`` -> ``eot`` array -> ``eot_merge`` (one of each of the last three per survey) ->
    ``magfit`` -> ``sb`` array (per survey) -> ``sb_merge`` -> ``plots``. Each job waits for the previous ones with
    ``--dependency=afterok``, so a failed job stops the chain. A frozen copy of the case is written to ``case.yaml``
    in the output directory and used by the jobs.

    Parameters
    ----------
    case : Case
    dry_run : bool, default False
        Only write the scripts and print the sbatch commands.
    from_stage : str, default 'prep'
        First stage to submit; earlier stages are assumed done and not waited for.
    skip : sequence of str, optional
        Stages to leave out.
    after : sequence of str, optional
        Extra job ids that the first `eot` jobs must wait for.

    Raises
    ------
    ValueError
        If `from_stage` is not a stage name.
    """
    if from_stage not in STAGE_ORDER:
        raise ValueError(f'from_stage must be one of {STAGE_ORDER}')
    start = STAGE_ORDER.index(from_stage)
    yaml_path = case.path('case.yaml')
    with open(yaml_path, 'w') as f:  # frozen copy of the config used for this run
        yaml.safe_dump(dataclasses.asdict(case), f)
    arg = yaml_path
    go = _Submitter(dry_run)

    def run(stage):
        """Whether a stage is part of this submission (not before `from_stage` and not skipped)."""
        return STAGE_ORDER.index(stage) >= start and stage not in skip

    def deps(*ids):
        """Drop the stages that were not submitted (None) from a list of job ids."""
        return [i for i in ids if i]  # None = stage skipped, nothing to wait for

    prep = go(_script(case, 'prep', f'prep {arg}', case.time_short, mem='16G', full_stack=True)) if run('prep') else None
    merge_ids = []
    for tel in case.telescopes:
        ini = initial(tel)
        f = e = m = None
        if run('fof'):
            f = go(_script(case, f'fof_{ini}', f'fof {arg} --tel {tel}', case.time_long, mem=case.fof_mem), after=deps(prep))
        if run('eot'):
            e = go(_script(case, f'eot_{ini}', f'eot {arg} --tel {tel}', case.time_long,
                           array=f'0-{case.n_eot_tasks - 1}'), after=deps(f, *after))
        if run('eot_merge'):
            m = go(_script(case, f'eotmerge_{ini}', f'eot_merge {arg} --tel {tel}', case.time_short), after=deps(e))
        merge_ids.append(m)
    magfit = None
    if run('magfit'):
        magfit = go(_script(case, 'magfit', f'magfit {arg}', case.time_short, cpus=4), after=deps(prep, *merge_ids))
    sb_ids = []
    for tel in case.telescopes:
        ini = initial(tel)
        if run('sb'):
            sb_ids.append(go(_script(case, f'sb_{ini}', f'sb {arg} --tel {tel}', case.time_long,
                                     array=f'0-{case.n_sb_tasks - 1}'), after=deps(magfit)))
    merge = None
    if run('sb_merge'):
        merge = go(_script(case, 'sbmerge', f'sb_merge {arg}', case.time_short), after=deps(*sb_ids))
    if run('plots'):
        go(_script(case, 'plots', f'plots {arg}', case.time_short), after=deps(merge))
