import functools

import numpy as np
from validphys.api import API
from validphys.loader import Loader
from validphys.fkparser import load_fktable

#  from validphys.convolution import OP

BASELINE_PDF = "220209-01-rs-nnpdf40"

config = {
    "fit": BASELINE_PDF,
    "use_t0": True,
    "use_cuts": "fromfit",
    "theory": {"from_": "fit"},
    "theoryid": {"from_": "theory"},
    "datacuts": {"from_": "fit"},
    "t0pdfset": {"from_": "datacuts"},
    "pdf": {"from_": "fit"},
    "dataset_inputs": {"from_": "fit"},
}


@functools.cache
def data():
    data = API.dataset_inputs_loaded_cd_with_cuts(**config)

    return data


def values():
    values = []

    for ds in data():
        values.append(ds.central_values.values)

    return np.concatenate(values)


@functools.cache
def covmat():
    cov = API.dataset_inputs_covmat_from_systematics(**config)

    return cov


@functools.cache
def theory():
    """Returns the fktable as a dense numpy array that can be directly
    manipulated with numpy
    The return shape is:
        (ndata, nx, nbasis) for DIS
        (ndata, nx, nx, nbasis) for hadronic
    where nx is the length of the xgrid
    and nbasis the number of flavour contributions that contribute
    """
    theoryid = API.fit(fit=config["fit"]).as_input()["theory"]["theoryid"]

    theory = []
    loader = Loader()

    for ds in data():
        spec = loader.check_dataset(ds.setname, theoryid=theoryid)

        cuts = spec.cuts.load()

        fkdata = dict(op=spec.op)
        fkdata["elements"] = []

        for fkspec in spec.fkspecs:
            fk = load_fktable(fkspec).with_cuts(cuts)
            # Make the dataframe into a dense numpy array
            df = fk.sigma

            # Read up the shape of the output table
            ndata = fk.ndata
            nx = len(fk.xgrid)
            nbasis = df.shape[1]

            if ndata == 0:
                if fk.hadronic:
                    return np.zeros((ndata, nbasis, nx, nx))
                return np.zeros((ndata, nbasis, nx))

            # First get the data index out of the way
            # this is necessary because cuts/shifts and for performance reasons
            # otherwise we will be putting things in a numpy array in very awkward orders
            ns = df.unstack(level=("data",), fill_value=0)
            x1 = ns.index.get_level_values(0)

            if fk.hadronic:
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

            fkdata["elements"].append(fktable)

        theory.append(fkdata)

    return theory
