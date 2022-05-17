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
    theoryid = API.fit(fit=config["fit"]).as_input()["theory"]["theoryid"]

    theory = []
    loader = Loader()

    for ds in data():
        spec = loader.check_dataset(ds.setname, theoryid=theoryid)

        cuts = spec.cuts.load()

        fkdata = dict(op=spec.op)
        fkdata["elements"] = []

        for fk in spec.fkspecs:
            df = load_fktable(fk).with_cuts(cuts).sigma
            shape = tuple(
                [
                    df.index.get_level_values(n).max() + 1
                    for n in range(len(df.index[0]))
                ]
                + [max(df.columns) + 1]
            )

            fkarray = np.zeros(shape)
            for el in df.iloc:
                for i, v in el.items():
                    fkarray[(*el.name, i)] = v

            fkdata["elements"].append(fkarray)

        theory.append(fkdata)

    return theory
