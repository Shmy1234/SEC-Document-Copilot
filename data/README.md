# Data

Local data artifacts for development live here.

- `downloads/` holds raw source files fetched from SEC EDGAR, grouped by year.
- `Markdown/` holds Docling-generated Markdown copies of the downloaded filings,
  grouped by the same year folders.
- `html_to_markdown/` holds the conversion script and tests.
- Downloaded payloads are gitignored because the corpus can get large.
- Fetch from the repository root with
  `SEC_USER_AGENT="Document Copilot you@example.com" uv run data/download.py`.
- Convert downloaded HTML filings from the repository root with
  `uv run --directory backend python ../data/html_to_markdown/convert_html_to_markdown.py --overwrite`.
- Preview chunking with
  `uv run --directory backend document-copilot-ingest --dry-run --limit-documents 1`.
- Upload one chunk first with
  `uv run --directory backend document-copilot-ingest --upload --limit-documents 1 --limit-chunks 1`.
- Upload the full corpus only after the smoke test with
  `uv run --directory backend document-copilot-ingest --upload --yes`.
