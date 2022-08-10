# -*- coding: utf-8 -*-
import pathlib

import click

from . import base


@base.command.group("report")
@click.option(
    "-d",
    "--destination",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "reports",
    help="Alternative destination path (default: $PWD/reports)",
)
def subcommand():
    """Generate fit reports"""
