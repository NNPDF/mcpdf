from typing import Union

import lhapdf
import numpy as np

XGRID_SIZE: int = 50


def xgrid(pdf: lhapdf.PDF, grid_size: int) -> np.ndarray:
    return np.geomspace(pdf.xMin, pdf.xMax, grid_size)


def xs(
    pdf: lhapdf.PDF, size: int, grid_size: int = XGRID_SIZE, index: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    xg = xgrid(pdf, grid_size)

    indices = (np.random.normal(scale=0.2, size=size) + 1.3) * grid_size / 2
    indices = np.minimum(np.maximum(0, indices), grid_size - 2)
    indices = indices.astype(np.int64)

    if index:
        return xg[indices], indices

    return xg[indices]


def flavors(pdf: lhapdf.PDF) -> np.ndarray:
    return np.array(pdf.info().get_entry("Flavors"))


def pids(
    pdf: lhapdf.PDF, size: int, index: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    flavs = flavors(pdf)
    indices = np.random.choice(len(flavs), size)

    if index:
        return np.array(flavs)[indices], indices

    return np.array(flavs)[indices]


def replicas(pdfs: list[lhapdf.PDF], size: int, Q2: float) -> np.ndarray:
    pdf = pdfs[0]

    pdf_replicas = np.array(pdfs)[np.random.choice(len(pdfs), size)]
    x_replicas = xs(pdf, size)
    pid_replicas = pids(pdf, size)

    data = []
    for pdfrep, x, pid in zip(pdf_replicas, x_replicas, pid_replicas):
        data.append(pdfrep.xfxQ2(pid, x, Q2))

    return np.stack([x_replicas, pid_replicas, data])


def hessian(pdfs: list, size: int, Q2: float) -> np.ndarray:
    pdf = pdfs[0]

    x_replicas, x_indices = xs(pdf, size, index=True)
    pid_replicas, pid_indices = pids(pdf, size, index=True)

    values = []
    for pdfrep in pdfs:
        pdfvals = []
        for fl in flavors(pdf):
            flavor = []
            for x in xgrid(pdf, XGRID_SIZE):
                flavor.append(pdfrep.xfxQ2(fl, x, Q2))
            pdfvals.append(flavor)
        values.append(pdfvals)

    values = np.array(values)
    weights = np.random.normal(size=size * len(pdfs)).reshape((size, len(pdfs)))
    sliced_values = np.moveaxis(values, 0, -1)[pid_indices, x_indices]
    data = np.einsum("ij,ij->i", sliced_values, weights)

    return np.stack([x_replicas, pid_replicas, data])


def load(name: str, size: int = 1000) -> np.ndarray:
    pdfs = lhapdf.mkPDFs(name)

    Q02 = pdfs[0].q2Min

    error_type = pdfs[0].info().get_entry("ErrorType")
    if error_type == "replicas":
        return replicas(pdfs, size, Q02)
    if error_type == "hessian":
        return hessian(pdfs, size, Q02)
    raise ValueError(f"PDF error type not recognized for '{name}'")
