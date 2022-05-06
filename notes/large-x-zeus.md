# Large x data from Allen Caldwell (ZEUS)

We have this dataset from Allen Caldwell, and we can implement into the fit.

## Because we can

- we have FkTables, and we can apply the transfer matrix to a binned integrated
  DIS, to obtain the full forward map as an FkTable
- MCPDF (HMC) will implement as the exact likelihood

### Comparisons

- GPPDF possibly without this dataset (if it's not possible to accommodate for
  it in `lsqfitgp`)
- NNPDF: we can implement with an approximation
  - pseudodata: sampled with actual Poisson
  - chi2: computed as usual (with variances from the mean, plus systematics)

## Because we actually want

- the large x data DIS, very much constraining the PDFs in a region with very
  little availability of other data
  - slide 20:
    https://indico.cern.ch/event/1072533/contributions/4813645/attachments/2439255/4178140/DIS22_caola.pdf
