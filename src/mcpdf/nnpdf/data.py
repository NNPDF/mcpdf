from typing import Sequence, Union

import numpy as np
from validphys.api import API

from . import defaults


def data(
    fit: str = defaults.BASELINE_PDF,
    use_t0: bool = True,
    use_cuts: str = "fromfit",
    theory: str = "fromfit",
    theoryid={"from_": "theory"},
    datacuts={"from_": "fit"},
    t0pdfset={"from_": "datacuts"},
    pdf={"from_": "fit"},
    dataset_inputs={"from_": "fit"},
):
    data = API.dataset_inputs_loaded_cd_with_cuts(
        fit=fit,
        use_t0=use_t0,
        use_cuts=use_cuts,
        theory=theory,
        theoryid=theoryid,
        datacuts=datacuts,
        t0pdfset=t0pdfset,
        pdf=pdf,
        dataset_inputs=dataset_inputs,
    )

    return data


def values(
    fit: str = defaults.BASELINE_PDF, dataset_inputs: Union[Sequence, str] = "fromfit"
):
    values = []

    for ds in data(fit=fit, dataset_inputs=dataset_inputs):
        values.append(ds.central_values.values)

    return np.concatenate(values)


def covmat(
    fit: str = defaults.BASELINE_PDF,
    use_t0: bool = True,
    use_cuts: str = "fromfit",
    theory={"from_": "fit"},
    theoryid={"from_": "theory"},
    datacuts={"from_": "fit"},
    t0pdfset={"from_": "datacuts"},
    pdf={"from_": "fit"},
    dataset_inputs={"from_": "fit"},
):
    cov = API.dataset_inputs_covmat_from_systematics(
        fit=fit,
        use_t0=use_t0,
        use_cuts=use_cuts,
        theory=theory,
        theoryid=theoryid,
        datacuts=datacuts,
        t0pdfset=t0pdfset,
        pdf=pdf,
        dataset_inputs=dataset_inputs,
    )

    return cov
