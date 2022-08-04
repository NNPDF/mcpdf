import pathlib

from validphys.api import API

runcards = pathlib.Path(__file__).parent / "runcards"


def report():
    print("Generating report")
