# -*- coding: utf-8 -*-
"""Create report to compare fits results."""
import datetime
import logging
import os
import pathlib
import shutil

import click

from ..fit import install, report
from . import base

_logger = logging.getLogger(__name__)


@base.command.command("report")
@click.argument("fit", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("--which", type=click.Choice(report.REPORTS), default=report.REPORTS[0])
@click.option("--runcard", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("--reinstall", is_flag=True)
@click.option(
    "-d",
    "--destination",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "reports",
    help="Alternative destination path (default: $PWD/reports)",
)
def subcommand(
    fit: pathlib.Path,
    which: str,
    runcard: pathlib.Path,
    reinstall: bool,
    destination: pathlib.Path,
):
    """Generate FIT comparison report.

    Produce reports following the template specified by runcard.

    """
    try:
        install.install(fit, force=reinstall)
    except FileExistsError:
        _logger.error(
            "The PDF set is already present, either rename or force reinstallation."
        )
        return

    tmpdir = report.ensure_runcard(runcard, which=which)
    os.environ["BROWSER"] = ""
    outdir = report.report(tmpdir)

    repdir = destination / f"{fit.name}"
    try:
        shutil.move(outdir, repdir)
    except shutil.Error:
        repdir = repdir.with_name(
            repdir.name + f"-{datetime.datetime.now().isoformat()}"
        )
        shutil.move(outdir, repdir)
    shutil.rmtree(tmpdir)

    _logger.info(f"Report generated, you can find in '{repdir}'")
