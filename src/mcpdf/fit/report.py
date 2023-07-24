# -*- coding: utf-8 -*-
"""Report fit result."""
import os
import pathlib
import shutil
import sys
import tempfile
from contextlib import contextmanager
from typing import Optional

import validphys.app as vp

REPORTS = ["base", "short", "full"]


def rcpath(which: str) -> pathlib.Path:
    """Path to pre-defined runcards.

    Parameters
    ----------
    which: str
        the

    Returns
    -------
    pathlib.Path
        path to runcard containing folder

    """
    return pathlib.Path(__file__).parent / f"runcard-{which}"


@contextmanager
def cwd(path: os.PathLike):
    """Switch current working directory.

    Parameters
    ----------
    path: os.PathLike
        the new path to which temporarily move the current working directory

    """
    oldpwd = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


@contextmanager
def argv(args: list[str]):
    """Switch command line arguments.

    Parameters
    ----------
    args: list[str]
        the new arguments to be temporarily used

    """
    oldargs = sys.argv.copy()
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = oldargs


def extract_runcard(which: str, dest: os.PathLike):
    """Extract runcard to destination.

    Parameters
    ----------
    which: str
        which runcard to extract
    dest: os.PathLike
        where to extract

    """
    for file in rcpath(which).glob("*"):
        shutil.copy2(file, dest)


def ensure_runcard(
    runcard: Optional[os.PathLike], which: str = REPORTS[0]
) -> pathlib.Path:
    """Generate wrapping folder with runcard.

    Parameters
    ----------
    runcard: None or os.PathLike
    which : str

    Returns
    -------
    pathlib.Path
        path to temporary folder, containing requested runcard

    """
    tmpdir = pathlib.Path(tempfile.mkdtemp())

    if runcard is None:
        extract_runcard(which, tmpdir)
    else:
        runcard = pathlib.Path(runcard)
        if runcard.is_file():
            shutil.copy2(runcard, tmpdir / "runcard.yaml")
        elif runcard.is_dir():
            for file in runcard.glob("*"):
                shutil.copy2(file, tmpdir)

    return tmpdir


def report(workdir: os.PathLike):
    """Produce comparison report.

    Parameters
    ----------
    workdir: os.PathLike
        path to folder containing report

    Returns
    -------
    pathlib.Path
        path to generated report folder

    """
    workdir = pathlib.Path(workdir)

    with cwd(workdir):
        with argv("validphys runcard.yaml".split()):
            vp.main()

    return workdir / "output"
