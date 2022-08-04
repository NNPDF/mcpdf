# Installation instructions

Updated instructions to install `vp` and make everything working.

1. Install `poetry`
2. Install the environment with `poetry install`
3. Follow instructions at NNPDF/nnpdf#1564 to install `vp`
   - as `PREFIX` use
     ```sh
     export PREFIX=$(poetry env info -p)
     ```

## Retrieve data

`vp` is mostly retrieving stuff on its own, but e.g. not for theory. So run:

```sh
vp-get theoryID 200
```

to install theory 200.

## Jupyter Notebook

Instructions to run Jupyter Notebooks in this repo.

0. Install the code (instructions above)
1. Install `poethepoet` (task runner for Poetry) - it has to be installed
   outside Poetry environment
2. Run `poe install-nb` to install the kernel relative to the Poetry environment
3. Launch notebook server (e.g. `poetry run jupyter lab`)
4. Select the installed kernel: "Kernel">"Change Kernel ...">"mcpdf-..."
