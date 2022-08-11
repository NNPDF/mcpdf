# -*- coding: utf-8 -*-
"""Evolve fit output."""
import functools
import logging
import os
import pathlib
import shutil
import tempfile

import eko
import ekobox as eb
import lhapdf
import numpy as np
import numpy.typing as npt
from ekobox import genpdf
from validphys import loader

_logger = logging.getLogger(__name__)

NREPLICAS = 10  # TODO: 100
NFLAVORS = 14
XGRID = np.geomspace(1e-09, 1.0, num=5)  # TODO: num=50
INITIAL_PDF = np.random.rand(NREPLICAS, NFLAVORS, XGRID.size)  # TODO: drop


def q2grid(Q0: float) -> npt.NDArray:
    """Create Q2 grid.

    Parameters
    ----------
    Q0: float
        initial scale

    Returns
    -------
    np.ndarray
        Q2 grid constructed

    """
    return np.geomspace(Q0**2, 1e10, num=6)  # TODO: num=100


@functools.cache
def theory_card(theoryid: int) -> dict:
    """Extract theory card from the database.

    Parameters
    ----------
    theoryid: int
        ID of the theory to be extracted

    Returns
    -------
    dict
        extracted theory card

    """
    theory = loader.Loader().check_theoryID(theoryid).get_description()
    theory.pop("FNS")
    # TODO: remove hardcoded PTO
    theory["PTO"] = 0
    return eb.gen_theory.gen_theory_card(theory["PTO"], theory["Q0"], update=theory)


@functools.cache
def operator_card(Q0: float) -> dict:
    """Generate suitable operator card.

    Parameters
    ----------
    Q0: float
        initial scale

    Returns
    -------
    dict
        generatd card

    """
    return eb.gen_op.gen_op_card(q2grid(Q0), update=dict(interpolation_xgrid=XGRID))


def evolve(theoryid: int) -> npt.NDArray:
    """Evolve initial PDF.

    Parameters
    ----------
    theoryid: int
        ID of the theory to be extracted

    Returns
    -------
    np.ndarray
        evolved PDF set, dimensions ``(rep, Q2, fl, x)``

    """
    central = INITIAL_PDF.mean(axis=0)
    initpdf = np.vstack((central[np.newaxis, :, :], INITIAL_PDF))

    tc = theory_card(theoryid)
    oc = operator_card(tc["Q0"])

    _logger.info("Computing evolution...")
    operator = eko.run_dglap(tc, oc)

    evolved = []
    for q2, op in operator["Q2grid"].items():
        _logger.info(f"Evolving {q2}")
        evolved.append(np.einsum("aibj,nbj->nai", op["operators"], initpdf))

    return np.transpose(np.array(evolved), (1, 0, 2, 3))


def dump(theoryid: int, evolved: npt.NDArray) -> pathlib.Path:
    """Dump an evolved PDF set in LHAPDF format.

    Parameters
    ----------
    theoryid: int
        ID of the theory to be extracted
    evolved: np.ndarray
        evolved PDF set

    Returns
    -------
    pathlib.Path
        path to temporary folder, containing the LHAPDF set

    """
    tc = theory_card(theoryid)
    oc = operator_card(tc["Q0"])

    tmpdir = tempfile.mkdtemp()
    pdfdir = pathlib.Path(tmpdir).absolute()

    info = eb.gen_info.create_info_file(tc, oc, NREPLICAS + 1, info_update={})
    for key, value in info.items():
        if isinstance(value, float):
            info[key] = float(value)
    genpdf.export.dump_info(pdfdir, info)

    block = dict(
        Q2grid=q2grid(tc["Q0"]).tolist(), pids=list(range(14)), xgrid=XGRID.tolist()
    )
    # data are x*pdf
    xgrid = oc["interpolation_xgrid"]
    xevolved = xgrid[np.newaxis, np.newaxis, np.newaxis, :] * evolved
    for idx, xreplica in enumerate(xevolved):
        block["data"] = xreplica.transpose(0, 2, 1).reshape(-1, xreplica.shape[1])
        genpdf.export.dump_blocks(
            pdfdir,
            idx,
            [block],
            pdf_type="PdfType: replica\nFromMCReplica: {idx}\n",
        )

    return pdfdir


def install(pdf: os.PathLike) -> pathlib.Path:
    """Install pdf in LHAPDF path.

    Parameters
    ----------
    pdf: os.PathLike
        path to pdf directory

    Returns
    -------
    str
        destination path

    """
    pdf = pathlib.Path(pdf)
    return shutil.copytree(pdf, pathlib.Path(lhapdf.paths()[0]) / pdf.name)
