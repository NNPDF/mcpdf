import json
from . import pdfs
import numpy as np
import matplotlib.pyplot as plt


def sample_GP(mean, cov, nsamples=1000):
    samples = np.random.multivariate_normal(mean, cov, size=nsamples)
    cv = samples.mean(axis=0)
    std = samples.std(axis=0)

    return cv, std


def plot_GPs_and_pdf(
    means,
    stds,
    labels,
    input_pdf,
    grids,
    flavor,
    mu,
    log_scale=False,
    ylinear=None,
    ylog=None,
):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))  # 1 row, 2 columns
    # fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,10))  # 1 row, 2 columns
    for cv, std, label, grid in zip(means, stds, labels, grids):

        # ax1.errorbar(grid,cv,fmt=".",yerr=std,linestyle='None')
        ax1.plot(grid, cv, "-")
        ax1.fill_between(grid, cv - std, cv + std, label=label, alpha=0.5)
        # ax2.errorbar(grid,cv,fmt=".",yerr=std,linestyle='None')
        ax2.plot(grid, cv, "-")
        ax2.fill_between(grid, cv - std, cv + std, label=label, alpha=0.5)

    grid = grids[0]  # to plot the pdf use the first available grid
    pdf = pdf_values(grid, input_pdf, mu, flavor)
    ax1.plot(grid, pdf)
    ax2.plot(grid, pdf)
    ax1.set_xscale("linear")
    ax2.set_xscale("log")
    ax1.set_ylim(ylinear)
    ax2.set_ylim(ylog)

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()


def plot_data_th_comparison(means, covs, labels, y, sys, stat):

    x = np.arange(y.size)

    # data
    # sys1 = sys['CONTLIMIT'].to_numpy()
    # sys2 = sys['FINITEVOLUME'].to_numpy()
    sys1 = sys[:, 0]
    sys2 = sys[:, 1]
    yerr = np.sqrt(sys1**2 + sys2**2 + stat**2)

    # pdf error
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.errorbar(x, y, yerr=yerr, fmt=".", label="data", color="green")

    for mean, cov, label in zip(means, covs, labels):
        pdf_error = np.sqrt(np.diag(cov))
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - pdf_error, mean + pdf_error, alpha=0.5)

    plt.legend()
    plt.show()
