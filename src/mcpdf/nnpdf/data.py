import functools

import numpy as np
from validphys.api import API

from . import defaults


@functools.cache
def data(config=defaults.config):
    data = API.dataset_inputs_loaded_cd_with_cuts(**config)

    return data


def values(config=defaults.config):
    values = []

    for ds in data(config):
        values.append(ds.central_values.values)

    return np.concatenate(values)


@functools.cache
def covmat(config=defaults.config):
    cov = API.dataset_inputs_covmat_from_systematics(**config)

    return cov
