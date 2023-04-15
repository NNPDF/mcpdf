import lhapdf
from collections.abc import Iterable
import math
import numpy as np


# flavor to evolution basis rotation
# (Sigma, g, V, V3, V8, V15, T3, T8, T15)
# = rot_to_evolution @ (g, u, d, s, c, ubar, dbar, sbar, cbar)

flavors = [21, 2, 1, 3, 4, -2, -1, -3, -4]

rot_to_evolution = np.asarray(
    [
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        [0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, -2.0, 0.0, -1.0, -1.0, 2.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, -3.0, -1.0, -1.0, -1.0, 3.0],
        [0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, -2.0, 0.0, 1.0, 1.0, -2.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, -3.0, 1.0, 1.0, 1.0, -3.0],
    ]
)


def xf(flavor, pdf_set, pdf_member, x, mu):
    pdf = lhapdf.mkPDF(pdf_set, pdf_member)
    if isinstance(x, Iterable):
        res = [pdf.xfxQ(flavor, xval, mu) for xval in x]
        res = np.asarray(res)
    else:
        res = pdf.xfxQ(flavor, x, mu)
    return res


def get_evolution_pdfs(pdf_set, pdf_member, x, mu):
    def get_flavor_xpdfs(pdf_set, pdf_member, x, mu):
        pdf_flavs = []
        for flav in flavors:
            pdf_flavs.append(xf(flav, pdf_set, pdf_member, x, 1.65))
        return np.asarray(pdf_flavs)

    pdf_flavs = get_flavor_xpdfs(pdf_set, pdf_member, x, mu) / x
    return np.reshape(rot_to_evolution @ pdf_flavs, (x.size * len(flavors)))

def get_evolution_xpdfs(pdf_set, pdf_member, x, mu):
    def get_flavor_xpdfs(pdf_set, pdf_member, x, mu):
        pdf_flavs = []
        for flav in flavors:
            pdf_flavs.append(xf(flav, pdf_set, pdf_member, x, 1.65))
        return np.asarray(pdf_flavs)

    pdf_flavs = get_flavor_xpdfs(pdf_set, pdf_member, x, mu)
    return np.reshape(rot_to_evolution @ pdf_flavs, (x.size * len(flavors)))
