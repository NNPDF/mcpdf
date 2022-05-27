from typing import Optional

import numpy as np
import pandas as pd
from validphys.api import API
from validphys.loader import Loader
from validphys.fkparser import load_fktable

#  from validphys.convolution import OP

from . import data, defaults


class FkTable:
    def __init__(self, table: np.ndarray, xgrid: np.ndarray, flavors: np.ndarray):
        self.table = table
        self.xgrid = xgrid
        self.flavors = flavors

    @classmethod
    def from_vp(cls, loaded):
        """Initialize from a `validphys` loaded FkTable.

        Parameters
        ----------

        """
        ndata = loaded.ndata
        nx = len(loaded.xgrid)
        table = cls.df_to_array(loaded.sigma, ndata, nx, hadronic=loaded.hadronic)

        return cls(table, xgrid=loaded.xgrid, flavors=loaded.sigma.columns.to_numpy())

    @staticmethod
    def df_to_array(
        df: pd.DataFrame, ndata: int, nx: int, hadronic: bool
    ) -> np.ndarray:
        """Make the dataframe into a dense numpy array

        Parameters
        ----------
        """
        # Read up the shape of the output table
        nbasis = df.shape[1]

        if ndata == 0:
            if hadronic:
                return np.zeros((ndata, nbasis, nx, nx))
            return np.zeros((ndata, nbasis, nx))

        # First get the data index out of the way
        # this is necessary because cuts/shifts and for performance reasons
        # otherwise we will be putting things in a numpy array in very awkward orders
        ns = df.unstack(level=("data",), fill_value=0)
        x1 = ns.index.get_level_values(0)

        if hadronic:
            x2 = ns.index.get_level_values(1)
            fkarray = np.zeros((nx, nx, ns.shape[1]))
            fkarray[x2, x1, :] = ns.values

            # The output is (ndata, basis, x1, x2)
            fktable = fkarray.reshape((nx, nx, nbasis, ndata)).T
        else:
            fkarray = np.zeros((nx, ns.shape[1]))
            fkarray[x1, :] = ns.values

            # The output is (ndata, basis, x1)
            fktable = fkarray.reshape((nx, nbasis, ndata)).T

        return fktable

    @property
    def ndata(self):
        return self.table.shape[0]


class FkCompound:
    def __init__(
        self,
        operation: str,
        elements: Optional[list[FkTable]] = None,
        name: Optional[str] = None,
    ):
        self.operation = operation
        self.elements = elements if elements is not None else []
        self.name = name

    def append(self, elem: FkTable):
        self.elements.append(elem)


def theory(fit: str = defaults.BASELINE_PDF, dataset_inputs=None, theoryid=None):
    """Returns the fktable as a dense numpy array that can be directly
    manipulated with numpy
    The return shape is:
        (ndata, nx, nbasis) for DIS
        (ndata, nx, nx, nbasis) for hadronic
    where nx is the length of the xgrid
    and nbasis the number of flavour contributions that contribute
    """
    if theoryid is None:
        theoryid = API.fit(fit=fit).as_input()["theory"]["theoryid"]

    theory = []
    loader = Loader()

    for ds in data.data():
        # skip unrequested datasets
        if dataset_inputs is not None:
            if ds.setname not in dataset_inputs:
                continue

        spec = loader.check_dataset(ds.setname, theoryid=theoryid)

        cuts = spec.cuts.load()

        fk_compound = FkCompound(name=spec.name, operation=spec.op)
        for fkspec in spec.fkspecs:
            loaded = load_fktable(fkspec).with_cuts(cuts)
            fk_compound.append(FkTable.from_vp(loaded))

        theory.append(fk_compound)

    return theory
