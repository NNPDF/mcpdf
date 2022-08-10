# -*- coding: utf-8 -*-
from . import base


@base.command("evolve")
def subcommand():
    """Evolve fit"""

    print("Evolved!")
