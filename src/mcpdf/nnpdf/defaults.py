BASELINE_PDF = "220209-01-rs-nnpdf40"

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
