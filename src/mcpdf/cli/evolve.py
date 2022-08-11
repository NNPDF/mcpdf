# -*- coding: utf-8 -*-
"""Evolve the result of a fit."""
import hashlib
import logging
import pathlib
import shutil

import click

from ..fit import evolve
from . import base

_logger = logging.getLogger(__name__)


@base.command.command("evolve")
@click.argument("theory", type=click.INT)
@click.option(
    "-d",
    "--destination",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "fits",
    help="Alternative destination path (default: $PWD/fits)",
)
def subcommand(theory: int, destination: pathlib.Path):
    """Evolve fit.

    Use THEORY (specified as the NNPDF `theory ID`_) to evove given PDF border
    condition at the given scale.

    .. _theory ID: https://docs.nnpdf.science/theory/theoryindex.html

    """
    pdf = evolve.evolve(theory)
    dest = (
        destination
        / f"mcpdf-{theory}-{hashlib.sha256(evolve.INITIAL_PDF).hexdigest()[:10]}"
    )

    pdfdir = evolve.dump(theory, pdf)
    shutil.move(pdfdir, dest)
    evolve.update_prefix(dest)

    _logger.info(f"Fit evolved, stored in '{dest}'")


@base.command.command("install")
@click.argument("pdf", type=click.Path(exists=True, path_type=pathlib.Path))
def install_subcommand(pdf):
    """Install PDF set.

    The PDF path given is consider to be the path to a folder containing info
    and replicas files.
    The folder is simply copied into a path discoverable by LHAPDF.

    """
    try:
        dest = evolve.install(pdf)
    except FileExistsError:
        _logger.error(f"'{pdf.name}' already installed")
        return

    _logger.info(f"PDF '{dest.name}' installed in '{dest.parent}'")
