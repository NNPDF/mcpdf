import functools

import numpy as np
from validphys.api import API

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
