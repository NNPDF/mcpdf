# Source

- `frompdf`, consider as data points directly an LHAPDF distribution (central
  value + uncertainty to generate a multi-Gaussian)
  - data loading -> `lhapdf`
  - no theory (it's just a delta)
  - simple comparison
- `frompseudodis`, consider some fake DIS data, generated à la Closure Tests
  (underlying PDF, known theory)
  - data loading -> `lhapdf` + theory
  - pseudo-theory: whatever theory we use to generate will enter in the
    likelihood
  - again simple comparison: we still have an underlying PDF
- `fromdis`, load actual DIS data
  - it will require validphys: central values + experimental covariance matrix
    - doable the standalone loading of central values, but covariances might be
      complicated to construct manually
