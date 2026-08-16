# Contributing

Contributions are welcome, especially independent validation cases, material
provenance improvements, numerical-stability fixes, and reproducible examples.

1. Fork the repository and create a focused branch.
2. Install with `python -m pip install -e ".[test]"`.
3. Add or update tests for every physics or contract change.
4. Run `python -m pytest -q` before opening a pull request.
5. Explain numerical conventions, units, validity limits, and data provenance.

Do not hide non-finite values, clip raw spectra to make a test pass, silently
extrapolate optical constants, or weaken an acceptance check without a
scientific justification and regression case.

