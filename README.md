# MCPDF

Sample the PDF posterior distribution.

## First implementation

- **prior**
  - use ToyLH for central values
  - use half the PDF value as variance ($2\sigma$ will be positive, mild
    negativity allowed)

## Posterior distribution

### Gaussianity

The posterior of a Gaussian process is not necessarily Gaussian itself.

It's easiest to show for a single Gaussian parameter: if the likelihood it's
non-Gaussian in the parameter itself, the posterior will not be Gaussian any
longer.
For example: even with a Gaussian prior and a Gaussian likelihood, if the
parameter is the variance of the likelihood, the posterior is not Gaussian.
