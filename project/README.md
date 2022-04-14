# Project

The best roadmap would include a number of increasingly complex result, that we
can achieve as fast as possible, each time showing the effectiveness of the
method.

The main idea is to solve issues one by one, instead of tackling immediately the
full complexity of the problem.

## From PDF

Central value + uncertainty to generate a multi-Gaussian likelihood.

- data loading -> `lhapdf`
- no theory (it's just a delta)
- simple comparison

## From Pseudo DIS

Closure Tests like: underlying PDF, known theory.

- data loading -> `lhapdf` + theory
- pseudo-theory: whatever theory we use to generate will enter in the
  likelihood
- again simple comparison: we still have an underlying PDF

## From DIS

- it will require validphys: central values + experimental covariance matrix
  - doable the standalone loading of central values, but covariances might be
    complicated to construct manually
