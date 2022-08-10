# -*- coding: utf-8 -*-
import pathlib

import click

from . import base
from ..fit import evolve


@base.command.command("evolve")
@click.option(
    "-d",
    "--destination",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=pathlib.Path.cwd().absolute() / "fits",
    help="Alternative destination path (default: $PWD/fits)",
)
def subcommand(destination: pathlib.Path):
    """Evolve fit"""

    pdf = evolve.evolve(200)
    evolve.dump(200, pdf, destination)
    print("Evolved!")
