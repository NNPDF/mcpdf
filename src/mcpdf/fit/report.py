# -*- coding: utf-8 -*-
"""Report fit result."""
import os
import pathlib
import shutil
import sys
import tempfile
from contextlib import contextmanager

import validphys.app as vp

runcards = pathlib.Path(__file__).parent / "runcards"


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


def report(runcard: os.PathLike) -> pathlib.Path:
    """Produce comparison report.

    Parameters
    ----------
    runcard: os.PathLike
        path to runcard to use for comparison

    Returns
    -------
    pathlib.Path
        path to generated report folder

    """
    runcard = pathlib.Path(runcard)

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    shutil.copy2(runcard, tmpdir / runcard.name)

    with cwd(tmpdir):
        with argv(f"vp {runcard.name}".split()):
            vp.main()

    return tmpdir / "output"
