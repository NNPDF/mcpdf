# Gaussianity

The posterior of a Gaussian process (including multi-Gaussian distribution) is
not necessarily Gaussian itself.

It's easiest to show for a single Gaussian parameter: if the likelihood it's
non-Gaussian in the parameter itself, the posterior will not be Gaussian any
longer.

Even with a Gaussian prior and a Gaussian likelihood, if the parameter is the
variance of the likelihood, the posterior is not Gaussian.

See [Non-Gaussian notebook](../experiments/non-gaussian.ipynb).
