# Document ingestion: vision models vs OCR - findings

**Prepared for executive review · July 14, 2026 · llm-bench evaluation lab**

## The question

We ingest business documents (invoices, statements, reports, forms) and need the
data inside them. The classic pipeline runs OCR over each page and hands the text
to software or a language model. Vision language models (VLMs) instead read the
page image directly. Is the VLM approach worth adopting, and which model should
read our pages?

## The one-paragraph answer

On pages that are plain printed text, OCR and vision tie - both captured 100% of
the values on our invoice set. The moment the answer lives in **visual structure**
(which bar is tallest, which box is checked, which price sits under which plan,
what a stamp says), OCR-based reading loses 17 to 29 points of accuracy while the
vision models keep reading at 93 to 100%. Since real document intake always
contains some tables, charts, forms, and stamps, **a vision model is the justified
default for document ingestion**, and the OCR pipeline is only defensible for
prose-only archives. The recommended reader is **Qwen2.5-VL 7B**: it read 100% of
all 50 ground-truth fields at about 2 seconds per page on a single NVIDIA L4.

## How we measured

- **Test set:** 15 synthetic business pages we authored with exact known answers -
  5 invoices/receipts, 5 table/chart pages, 5 forms/records. 50 graded fields in
  total (totals, dates, IDs, chart maxima, checked boxes, stamp status).
- **The isolating comparison:** every model read every page **twice** - once as
  the page image (the vision path) and once as OCR text from RapidOCR (the classic
  pipeline). Same model, same question, same deterministic per-field grading, so
  the difference is purely the ingestion modality. No judge model anywhere.
- **Hardware:** all local. Gemma 3 27B on a Dell GB10; Qwen2.5-VL 7B and
  Phi-3.5 Vision 4B each on one NVIDIA L4 (Dell XR7620) under vLLM.

## Results

### Field accuracy, page image vs OCR text

| Document set | Gemma 3 27B | Qwen2.5-VL 7B | Phi-3.5 Vision 4B |
|---|---|---|---|
| Invoices & receipts (18 fields) | 100% vs 100% | 100% vs 100% | 100% vs 100% |
| Tables & charts (14 fields) | **100% vs 79%** | **100% vs 71%** | **93% vs 79%** |
| Forms & records (18 fields) | **100% vs 78%** | **100% vs 83%** | **100% vs 83%** |
| **All 50 fields** | **100% vs 86%** | **100% vs 86%** | **98% vs 88%** |

Reading: "100% vs 79%" means the model read 100% of fields correctly from the
page image, but only 79% when the same model worked from OCR text alone.

### Where OCR fails, concretely

- **Bar chart:** every model named the highest and lowest sales months from the
  image; from OCR text every model got **0 of 2** - the text contains the month
  names and axis numbers but not the bar heights.
- **Checkbox form:** which two services were ticked and which priority was
  selected survive in the image only; OCR text lists all options with no tick state.
- **Stamped memo:** the diagonal APPROVED stamp degrades or vanishes in OCR text.
- **Pricing grid:** OCR flattens the table, so "the Business plan's price" loses
  its column association.

This matches NVIDIA's published RAG Blueprint accuracy benchmarks, where the VLM
ingestion path beats text-only retrieval on visually complex sets (RagBattlepacket
0.867 vs 0.812; consistent gains across Vidore domains) and ties on prose-dense
sets (DC767 ~0.90 either way).

### Speed and cost

| Model | Where it runs | Time per page | Pages per minute |
|---|---|---|---|
| Qwen2.5-VL 7B | 1x NVIDIA L4 | ~2.1 s | ~29 |
| Phi-3.5 Vision 4B | 1x NVIDIA L4 | ~1.4 s | ~44 |
| Gemma 3 27B | Dell GB10 | ~20 s | ~3 |

All processing stayed on hardware we own - no per-page or per-token API spend.
A single L4 running Qwen2.5-VL clears roughly 1,700 pages an hour; two L4s in
this lab clear twice that.

## Recommendation

1. **Adopt the vision path for document intake.** Use **Qwen2.5-VL 7B** as the
   default reader (100% field accuracy, ~2 s/page on one L4). Keep
   **Phi-3.5 Vision** as the high-volume option where a ~2-point accuracy trade
   is acceptable for ~1.5x the throughput.
2. **Do not pay for a large model by default.** Gemma 3 27B read no better than
   the 7B on these pages and is 10x slower per page; hold it in reserve for
   pages the small readers miss.
3. **Keep an OCR lane only for prose-only archives**, where it tied at 100% and
   is the cheapest path.
4. **Validate on our own documents next.** The llm-bench dashboard now carries
   this exact evaluation as reusable Document ingestion cards; rerunning the
   comparison on a sample of real pages is a one-click exercise.

## Reproducing this in llm-bench

Settings > Models > "Document ingestion (vision)" arms the three vision models;
Load tasks > pick a Docs card; press Play. The Compare panel reruns every page
from OCR text automatically, and the Recommendation panel produces this
vision-vs-OCR table plus a downloadable report.

*Method notes: pages generated by `dashboard/docsets/generate.py` (deterministic,
values computed at render time); OCR text precomputed once with RapidOCR
(PaddleOCR models) and shared by all models; grading is normalized
substring match per field ($1,284.50 = 1284.50), no LLM judge. NVIDIA references:
docs.nvidia.com/rag accuracy benchmarks; developer.nvidia.com blog on chunking
strategy (page-level chunking, tables/charts preserved whole - the design this
one-page-per-row dataset follows).*
