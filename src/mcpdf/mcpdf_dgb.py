from nnpdf import data, defaults, theory
from inference import pdfs
import numpy as np
import matplotlib.pyplot as plt

datasets_list = [
    {'dataset': 'HERACOMB_SIGMARED_B', 'frac': 0.75},
    {'dataset': 'HERACOMB_SIGMARED_C', 'frac': 0.75}
]

# load data and covmat for all the dataset
y = data.values(fit=defaults.BASELINE_PDF, dataset_inputs=datasets_list)
cov = data.covmat(fit=defaults.BASELINE_PDF, dataset_inputs=datasets_list)

fks = theory.theory(dataset_inputs=[ds['dataset'] for ds in datasets_list])