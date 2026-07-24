# Task format: bring your own tasks as plain files

Bake-off deliberately has no task-builder application. A task is a plain file in
one of the formats below; anything that can produce a file can produce a task.
Load one by pasting it (or its GitHub/GitLab link) into the app, or with
Upload from computer. Review, then save it as a card.

## Text tasks (Q&A)

Any of these parse automatically:

- **CSV / TSV** with a question column (`question`, `prompt`, `input`, or `query`)
  and optionally an answer column (`answer`, `expected`, `expected_answer`,
  `correct`, or `contains`)
- **JSON / JSONL** with the same field names
- **Markdown tables**, **`Q:` / `A:` pairs**, or plain one-question-per-line text

With an answer column, grading is a case-insensitive contains check: the answer
appearing anywhere in the model's reply counts. Without one, the task is
open-ended and measures speed only.

```csv
question,answer
What is the capital of Australia?,Canberra
Who wrote the novel 1984?,Orwell
```

### Strict rules (optional)

For format discipline, extraction drills, or key-point grading, use columns
named `__expected` (or `__expected_1`, `__expected_2`, ...) instead of `answer`.
Each value may carry a rule prefix:

| Value | Passes when |
|---|---|
| `Canberra` or `icontains:Canberra` | the text appears anywhere in the reply |
| `equals:yes` | the whole reply, trimmed, is exactly this |
| `regex:(?i)^\W*positive\W*$` | the pattern matches the reply |

**Several `__expected` columns must ALL pass** - that is how you grade a summary
on whether every key fact survived, or require both a value and its format.

## Document tasks (vision models)

A document set is a folder the dashboard serves from `dashboard/docsets/<id>/`:

```
dashboard/docsets/my-docs/
  page-1.png            page images, any count
  page-2.png
  manifest.json
```

`manifest.json`:

```json
{"id": "my-docs", "name": "My documents", "desc": "What this set tests.",
 "type": "doc", "suggest": 512,
 "rows": [
   {"file": "page-1.png",
    "label": "Invoice - Acme Ltd",
    "question": "This is a scanned document. Extract these fields exactly as printed: Total due; Invoice date. Answer with one line per field in the form \"Field name: value\". If a field is not on the document, write \"not found\".",
    "fields": {"Total due": "$1,284.50", "Invoice date": "March 3, 2026"},
    "ocr": "optional: pre-extracted OCR text for the vision-vs-OCR compare"}
 ]}
```

Each field is graded on its exact value appearing in the reply (normalized, so
`$1,284.50` also matches `1284.50`). The optional `ocr` / `parse` strings feed
the "reading the page vs reading extracted text" comparison; bake them with
`docsets/ocr_precompute.py` (RapidOCR) so that comparison has real text to use.
The set appears in the task list after the dashboard restarts.

## Moving tasks between installs

Settings > General > Move setup exports every saved task (with machines and the
lineup) as one file; import it on any other Bake-off.
