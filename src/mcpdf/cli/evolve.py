# -*- coding: utf-8 -*-
import pathlib
import logging

import click

from . import base
from ..fit import evolve

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
    """Evolve fit

    Use THEORY (specified as the NNPDF `theory ID`_) to evove given PDF border
    condition at the given scale.

    .. _theory ID: https://docs.nnpdf.science/theory/theoryindex.html

    """

    pdf = evolve.evolve(theory)
    evolve.dump(theory, pdf, destination)
    _logger.info(f"Fit evolved, stored in '{destination}/mcpdf'")
