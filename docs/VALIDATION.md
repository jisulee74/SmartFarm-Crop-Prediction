# Source snapshot validation — 2026-10-29

- Core regression, catalog, stability, exhaustive, semantic-control, completion,
  screening and tomato pipeline tests: **59 passed** on Python 3.11.
- The catalog's two required metadata dictionaries are tracked under
  `AXData/ax_catalog_experiments/reference/`; no training data is needed to build the catalog.
- Changed Python sources, JSON and shell scripts were syntax checked.
- Changed files were checked for known local credentials and common key formats.
- Existing dashboard HTML/JS/CSS/data files match the fetched origin/main versions.
- Extended dashboard tests: 10 passed, 5 failed. Three require local experiment-output
  files outside the published layout; two expect final-completed semantic results while
  the current published bundle is still a provisional screening result. These results
  were preserved, not relabelled as completed to satisfy the tests.

No new training, final-results publication or Excel report generation was performed.
The full installed package list and detailed logs are in the private workspace backup.

## Actual GitHub clone and private restore

- Fresh GitHub clone: 59 synthetic tests passed with the recorded existing environment.
- Restore safety: 7 checks passed (no overwrite, traversal/escape rejection, checksum, aliases, delta precedence).
- Restored six formal parquet hashes matched. Rebuilt six base split files from frozen observations/context; byte hashes, counts, feature lists and facility split membership all matched the archived base.
- Two prepared-data aliases are restored as reviewed relative links inside the clone.
- See `reproducibility/clone_validation.json` for the exact tested source revision and limits.
