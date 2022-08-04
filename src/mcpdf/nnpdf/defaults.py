BASELINE_PDF = {200: "210713-n3fit-001", 400: "220731-jcm-th400"}

config = {
    "fit": BASELINE_PDF[200],
    "use_t0": True,
    "use_cuts": "fromfit",
    "theory": {"from_": "fit"},
    "theoryid": {"from_": "theory"},
    "datacuts": {"from_": "fit"},
    "t0pdfset": {"from_": "datacuts"},
    "pdf": {"from_": "fit"},
    "dataset_inputs": {"from_": "fit"},
}
