# -*- coding: utf-8 -*-
"""Create report to compare fits results."""
import logging
import pathlib

import click

from ..fit import report
from . import base

_logger = logging.getLogger(__name__)


@base.command.command("report")
@click.argument("runcard", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "-d",
    "--destination",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "reports",
    help="Alternative destination path (default: $PWD/reports)",
)
def subcommand(runcard: pathlib.Path, destination: pathlib.Path):
    """Generate fit reports.

    Produce reports following the template specified by RUNCARD.

    """
    repdir = report.report(runcard)

    _logger.info(f"Report generated, you can find in '{repdir}'")
