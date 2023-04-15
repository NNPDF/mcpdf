import numpy as np

eps = 1e-5
alpha = 0.

def l(x, l0):
    return l0 * (x + eps)


def f(x, l0, alpha=alpha):
    return l0 * (x + eps) ** (alpha + 1)


def df(x, l0, alpha=alpha):
    return l0 * (alpha + 1.0) * (x + eps) ** alpha


### Gibbs kernel, representing T3
def k(x, y, sigma, l0):
    return (
        sigma**2
        * np.sqrt(2 * l(x, l0) * l(y, l0) / (l(x, l0) ** 2 + l(y, l0) ** 2))
        * np.exp(-((x - y) ** 2) / (l(x, l0) ** 2 + l(y, l0) ** 2))
    )


# kernel representing primitive of V3 (in notes and handwritten computations you called this Theta)
def K(x, y, sigma, l0, alpha=alpha):
    return f(x, l0, alpha) * k(x, y, sigma, l0) * f(y, l0, alpha)


# kernel representing V3
def dxdyK(x, y, sigma, l0, alpha=alpha):
    res = (
        1
        / (
            2
            * np.sqrt(2)
            * l0**2
            * (2 * eps**2 + x**2 + y**2 + 2 * eps * (x + y)) ** 4
        )
        * np.exp(-((x - y) ** 2 / (l0**2 * ((eps + x) ** 2 + (eps + y) ** 2))))
        * sigma**2
        * (eps + x) ** alpha
        * (eps + y) ** alpha
        * np.sqrt(
            ((eps + x) * (eps + y))
            / (2 * eps**2 + x**2 + y**2 + 2 * eps * (x + y))
        )
        * (
            32 * eps**8 * l0**2 * (2 + (3 + 2 * alpha * (2 + alpha)) * l0**2)
            + 128
            * eps**7
            * l0**2
            * (2 + (3 + 2 * alpha * (2 + alpha)) * l0**2)
            * (x + y)
            - 16 * x**2 * y**2 * (x**2 - y**2) ** 2
            + 16
            * eps**6
            * (
                (-4 + 18 * l0**2 + (45 + 32 * alpha * (2 + alpha)) * l0**4) * x**2
                + 2
                * (4 + 38 * l0**2 + 3 * (13 + 8 * alpha * (2 + alpha)) * l0**4)
                * x
                * y
                + (-4 + 18 * l0**2 + (45 + 32 * alpha * (2 + alpha)) * l0**4)
                * y**2
            )
            + 16
            * eps**5
            * (x + y)
            * (
                (-12 - 2 * l0**2 + (51 + 40 * alpha * (2 + alpha)) * l0**4) * x**2
                + 2
                * (12 + 58 * l0**2 + (33 + 16 * alpha * (2 + alpha)) * l0**4)
                * x
                * y
                + (-12 - 2 * l0**2 + (51 + 40 * alpha * (2 + alpha)) * l0**4)
                * y**2
            )
            - 16
            * l0**2
            * x
            * y
            * (x**2 + y**2)
            * (x**4 - 4 * x**2 * y**2 + y**4)
            + l0**4
            * (x**2 + y**2) ** 2
            * (
                (3 + 4 * alpha * (2 + alpha)) * x**4
                + 2 * (9 + 4 * alpha * (2 + alpha)) * x**2 * y**2
                + (3 + 4 * alpha * (2 + alpha)) * y**4
            )
            + 4
            * eps**4
            * (
                (-52 - 80 * l0**2 + 17 * (9 + 8 * alpha * (2 + alpha)) * l0**4)
                * x**4
                + 8
                * (-4 + 35 * l0**2 + (51 + 32 * alpha * (2 + alpha)) * l0**4)
                * x**3
                * y
                + 6
                * (28 + 120 * l0**2 + (93 + 56 * alpha * (2 + alpha)) * l0**4)
                * x**2
                * y**2
                + 8
                * (-4 + 35 * l0**2 + (51 + 32 * alpha * (2 + alpha)) * l0**4)
                * x
                * y**3
                + (-52 - 80 * l0**2 + 17 * (9 + 8 * alpha * (2 + alpha)) * l0**4)
                * y**4
            )
            + 8
            * eps**3
            * (x + y)
            * (
                (-12 - 36 * l0**2 + (39 + 40 * alpha * (2 + alpha)) * l0**4)
                * x**4
                + 8
                * (-4 + 7 * l0**2 + (9 + 4 * alpha * (2 + alpha)) * l0**4)
                * x**3
                * y
                + 2
                * (44 + 92 * l0**2 + (57 + 40 * alpha * (2 + alpha)) * l0**4)
                * x**2
                * y**2
                + 8
                * (-4 + 7 * l0**2 + (9 + 4 * alpha * (2 + alpha)) * l0**4)
                * x
                * y**3
                + (-12 - 36 * l0**2 + (39 + 40 * alpha * (2 + alpha)) * l0**4)
                * y**4
            )
            + 4
            * eps**2
            * (
                (-4 - 28 * l0**2 + (27 + 32 * alpha * (2 + alpha)) * l0**4) * x**6
                + 24
                * (-2 - 2 * l0**2 + (3 + 2 * alpha * (2 + alpha)) * l0**4)
                * x**5
                * y
                + 3
                * (-4 + 60 * l0**2 + (51 + 32 * alpha * (2 + alpha)) * l0**4)
                * x**4
                * y**2
                + 8
                * (16 + 30 * l0**2 + 3 * (7 + 4 * alpha * (2 + alpha)) * l0**4)
                * x**3
                * y**3
                + 3
                * (-4 + 60 * l0**2 + (51 + 32 * alpha * (2 + alpha)) * l0**4)
                * x**2
                * y**4
                + 24
                * (-2 - 2 * l0**2 + (3 + 2 * alpha * (2 + alpha)) * l0**4)
                * x
                * y**5
                + (-4 - 28 * l0**2 + (27 + 32 * alpha * (2 + alpha)) * l0**4)
                * y**6
            )
            + 8
            * eps
            * (x + y)
            * (
                -4 * x * (x - y) ** 2 * y * (x**2 + 4 * x * y + y**2)
                + l0**4
                * (x**2 + y**2)
                * (
                    (3 + 4 * alpha * (2 + alpha)) * x**4
                    + 3 * x**3 * y
                    + 4 * (3 + 2 * alpha * (2 + alpha)) * x**2 * y**2
                    + 3 * x * y**3
                    + (3 + 4 * alpha * (2 + alpha)) * y**4
                )
                - 2
                * l0**2
                * (
                    x**6
                    + 6 * x**5 * y
                    - 15 * x**4 * y**2
                    - 15 * x**2 * y**4
                    + 6 * x * y**5
                    + y**6
                )
            )
        )
    )
    return res

# derivative of the Gibbs kernel wrt first variable. Needed to compute dxK
def dxk(x, y, sigma, l0):
    return -(
        (
            np.exp(-((x - y) ** 2 / (l0**2 * ((eps + x) ** 2 + (eps + y) ** 2))))
            * sigma**2
            * (x - y)
            * (eps + y)
            * (2 * eps + x + y)
            * (
                2 * eps**2 * (2 + l0**2)
                + 4 * x * y
                + 2 * eps * (2 + l0**2) * (x + y)
                + l0**2 * (x**2 + y**2)
            )
        )
        / (
            np.sqrt(2)
            * l0**2
            * np.sqrt(
                ((eps + x) * (eps + y))
                / (2 * eps**2 + x**2 + y**2 + 2 * eps * (x + y))
            )
            * (2 * eps**2 + x**2 + y**2 + 2 * eps * (x + y)) ** 3
        )
    )


def dxK(x, y, sigma, l0, alpha=alpha):
    return df(x, l0, alpha) * k(x, y, sigma, l0) * f(y, l0, alpha) + f(x, l0, alpha) * dxk(
        x, y, sigma, l0
    ) * f(y, l0, alpha)