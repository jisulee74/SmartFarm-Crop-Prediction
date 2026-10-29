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
