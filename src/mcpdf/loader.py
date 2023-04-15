from nnpdf import data, defaults, theory
import numpy as np


# possible flavors appearing in DIS FK tables
flavs = [
    1,  # Sigma
    2,  # g
    3,  # V
    4,  # V3
    5,  # V8
    6,  # V15
    9,  # T3
    10,  # T8
    11,  # T15
]


def ordered_list(datasets_list):
    """Given an input list of datasets names,
    return a list of dictionaries of the kind
    [
        {'dataset': 'dataset_name'},
        ...
    ]
    following the ordering of data.data(), which is the one used
    when building a unique FK table
    """
    dataset_inputs = []
    for ds in data.data():
        ds_spec = {}
        if ds.setname in datasets_list:
            ds_spec["dataset"] = ds.setname
            dataset_inputs.append(ds_spec)
    return dataset_inputs


def load_data(datasets_list):
    """Load the selected experimental data in the order in which
    they will be loaded in the corresponding FK table"""

    # this makes sure that the order of the datapoints will be consistent
    # with the one in the FK table
    dataset_inputs = ordered_list(datasets_list)

    # load data and covmat for all the dataset
    y = data.values(fit=defaults.BASELINE_PDF, dataset_inputs=dataset_inputs)
    cov = data.covmat(fit=defaults.BASELINE_PDF, dataset_inputs=dataset_inputs)
    return y, cov


def build_FK(datasets_list, theoryid=None, xgrid):
    """Return a unique FK table for all the datasets"""

    fks = theory.theory(dataset_inputs=datasets_list, theoryid=theoryid)
    if theoryid is None:
        nxgrid = xgrid.size
    else:
        nxgrid = (fks[0].elements[0].xgrid).size

    # look at the flavors of loaded fk tables and build the corresponding boolean mask
    mask_list = []
    for fk in fks:
        mask = [
            flav in fk.elements[0].flavors for flav in flavs for k in range(0, nxgrid)
        ]
        mask_list.append(mask)

    # now build the corresponding matrices
    matrix_list = []
    for m in mask_list:
        matrix = []
        for pos, i in enumerate(m):
            if i:
                tmp = np.zeros(len(flavs) * nxgrid)
                tmp[pos] = 1.0
                matrix.append(tmp)

        matrix_list.append(np.array(matrix))

    # reinterpolate to common grid and rershape in 2 dim
    fk_bare_list = []
    for fk in fks:
        fk.x_reshape(xgrid)
        ndata, nbasis, nx = fk.elements[0].table.shape
        # xfk = np.multiply(fk.elements[0].table, xgrid)
        # fk_bare_list.append(np.reshape(xfk, (ndata, nbasis * nx)))
        fk_bare_list.append(np.reshape(fk.elements[0].table, (ndata, nbasis * nx)))

    # multiply fk for the matrix
    reshaped_fk = []
    for fk_bare, matrix in zip(fk_bare_list, matrix_list):
        reshaped_fk.append(fk_bare @ matrix)

    FK = np.concatenate([fk for fk in reshaped_fk], axis=0)

    return FK


#######################################


def build_FK_FNS(xgrid):
    """Build the fk table for the observable
    F^{NS} = F^p_2 - F^d_2, starting from the FK for F^p_2
    and taking as input the xgrid"""
    dataset = ["BCDMSP_dwsh"]
    FKp = build_FK(dataset, xgrid)
    indexT3 = 6
    lengrid = xgrid.size

    # FK_FNS_ = FKp[:,indexT3*lengrid:(indexT3+1)*lengrid]
    FKp[:, : indexT3 * lengrid] = 0.0
    FKp[:, (indexT3 + 1) * lengrid :] = 0.0

    FK_FNS = FKp[85:, :]  # remove first 85 datapoints
    return FK_FNS


def load_data_FNS():
    datasets = [
        {"dataset": "BCDMSP_dwsh", "frac": 0.75},
        {"dataset": "BCDMSD_dw_ite", "frac": 0.75},
    ]

    # load data and covmat for BCDMSP and BCDMSD
    y = data.values(fit=defaults.BASELINE_PDF, dataset_inputs=datasets)
    cov = data.covmat(fit=defaults.BASELINE_PDF, dataset_inputs=datasets)

    # first 85 datapoints correspond to those for Fp at 100 GeV,
    # which are those without a Fd counterpart
    Fp = y[85:333]
    Fd = y[333:]

    # construct cov for FpFp, FdFd, FpFd and FdFp
    covFpFp = cov[85:333, 85:333]
    covFdFd = cov[333:, 333:]
    covFpFd = cov[85:333, 333:]
    covFdFp = cov[333:, 85:333]

    # eq.(5) https://arxiv.org/pdf/hep-ph/0204232.pdf
    FNS = Fp - Fd
    covFNS = covFpFp + covFdFd - covFpFd - covFdFp

    return FNS, covFNS
