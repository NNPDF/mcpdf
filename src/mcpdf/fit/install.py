# -*- coding: utf-8 -*-
"""Install PDF set."""
import os
import pathlib
import shutil

import lhapdf


def install(pdf: os.PathLike, force: bool = False) -> pathlib.Path:
    """Install PDF in LHAPDF path.

    Parameters
    ----------
    pdf: os.PathLike
        path to pdf directory
    force: bool
        force reinstallation (thus does not fail if already existing)

    Returns
    -------
    str
        destination path

    Raises
    ------
    FileExistsError
        if PDF set already present

    """
    pdf = pathlib.Path(pdf).absolute()
    dest = pathlib.Path(lhapdf.paths()[0]) / pdf.name

    if force and dest.exists():
        shutil.rmtree(dest)

    return shutil.copytree(pdf, dest)
