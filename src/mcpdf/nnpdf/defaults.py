BASELINE_PDF = "210713-n3fit-001" # corresponding to NNPDF40 nnlo baseline fit

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
