# Extra

Extra improvement beyond the baseline sampling.

## Hyper-optimization

Still possible, maximazing the marginal likelihood (e.g. after a first rougher
sampling, that might be same size of the actual one, but in the enlarged space).

## Strong coupling

Inheriting theory from SIMUnet (i.e. FkTables interpolation), we could sample
the simultaneous distribution of PDF and $\alpha_s$.

The further dimension is not a theoretical problem, and only requires a
dedicated prior (that is simple to pick).
In practice, it might become a problem in the same way it is for SIMUnet, if the
underlying distribution is made weird by the inclusion of $\alpha_s$ (if it is
the optimization procedure instead, it might avoid the issue).

## Extrapolation

Since we have more explicit prior knowledge, whatever we get for the
extrapolation will be a direct consequence of our prior.

If the prior propagates no information on neighbor points, there would be no
interpolation as well.
If it propagates only the amount we need in interpolation, we might obtain a
very short extrapolation range (and we might advocate it's the correct thing).

Otherwise, we might choose a larger correlation length, and even decide to apply
it only on the extrapolation region: this is really prior knowledge, and we are
transparently including it or not in our sampling.

In any case, it will be explicit that is an arbitrary choice, as it has always
been also for the NN.

## Theory uncertainties:

It is possible a more direct implementation, independent from theory covmat:

$$
P(f | x) = \sum_T P(x | f, T) P(T) P(f)
$$

Such that we can attribute our favorite prior to the thoery, and avoid any
problematic detail (such that requiring exact Gaussianity of the distribution,
that is not the case already).
