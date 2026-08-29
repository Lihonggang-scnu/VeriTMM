# Expected-result policy

Expected capability, exact failure-code sets, artifacts, and small numerical
assertions live with each case under `cases/`. They are reviewed declarations,
not snapshots learned from the implementation under test.

The benchmark compares failure-code sets exactly. Unsupported physics is a
false acceptance whenever preflight reports ready. Numerical assertions are
limited to stable, interpretable properties; the normal physics verifier remains
the authority for a certificate.
