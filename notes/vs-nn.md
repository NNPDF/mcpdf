# Comparison with NN

Neural Networks (NN) are good when the problem is difficult to spell out
analytically.

- recognizing a cat in the wild is made of so many parts and relations between
  parts that is difficult for a person to spell out the relevant details
- a PDF is already an abstraction with some clean analytical properties:
  - mostly positive
  - sum rules
  - locality (interpolation)
  - behavior in extrapolation region
  - log small-x, linear large-x

The NN in the PDF case is hiding useful analytical details, and we're loosing
introspection that we had in the black box.
While there is nothing bad when you have very little introspection, the whole
set of properties is almost flushed out in the details of the NN.

Bayesian approach is not only better in the statistical treatment of the
involved distributions, but transparently propagates the analytical properties
of the prior into the posterior.
