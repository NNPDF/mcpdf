import lhapdf
import numpy as np


def xs(pdf: lhapdf.PDF, size: int, grid_size: int = 50) -> np.ndarray:
    xgrid = np.geomspace(pdf.xMin, pdf.xMax, grid_size)

    indices = (np.random.normal(scale=0.2, size=size) + 1.3) * grid_size / 2
    indices = np.minimum(np.maximum(0, indices), grid_size - 2)

    return xgrid[indices.astype(np.int64)]


def pids(pdf: lhapdf.PDF, size: int) -> np.ndarray:
    flavors = pdf.info().get_entry("Flavors")
    return np.array(flavors)[np.random.choice(len(flavors), size)]


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
    return np.zeros(size)


def load(name: str, size: int = 1000) -> np.ndarray:
    pdfs = lhapdf.mkPDFs(name)

    Q02 = pdfs[0].q2Min

    error_type = pdfs[0].info().get_entry("ErrorType")
    if error_type == "replicas":
        return replicas(pdfs, size, Q02)
    if error_type == "hessian":
        return hessian(pdfs, size, Q02)
    raise ValueError(f"PDF error type not recognized for '{name}'")
