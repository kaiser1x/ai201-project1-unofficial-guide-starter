# The Unofficial Guide — Project 1

A local-first Retrieval-Augmented Generation (RAG) system over real-estate documents: residential leases, property-inspection checklists, and regional property-tax guides. Ask a plain-language question; get an answer grounded strictly in the loaded documents, with source attribution.

---

## Domain and Document Sources

**Domain:** Real Estate Operations, Property Management, and Regional Property Taxation. This knowledge is valuable because the legal obligations of residential leases, the standards behind property maintenance/inspection, and hyper-local *ad valorem* tax rules are gatekept behind dense legal contracts, confusing government manuals, and fragmented invoice documentation. A unified RAG pipeline makes that fragmented data queryable for tenants, property managers, and homeowners.

**Sources** (files in `documents/`; two filenames differ from the original blueprint by a trailing descriptor — corrected here):

| # | Source File | Type | Scope / Region |
|---|-------------|------|----------------|
| 1 | `Lease-Agreement-Basic-Residential.pdf` | Lease Contract | Generic Residential |
| 2 | `Lease-Agreement-Simple-Form.pdf` | Lease Contract | Short-Form Generic |
| 3 | `Lease-Agreement-Standard-Residential.pdf` | Lease Contract | Deep Multi-Tenant |
| 4 | `Inspection-Checklist-HUD-HCV.pdf` | Inspection Form | HUD Section 8 (Housing Choice Voucher) |
| 5 | `Inspection-Checklist-Property-Management-Smartsheet.pdf` | Inspection Form | Smartsheet Template |
| 6 | `Inspection-Checklist-Total-Home-Inspection.pdf` | Inspection Form | Structural Diagnostics |
| 7 | `Property-Tax-Guide-Georgia-Citizens.pdf` | Government Guide | Georgia State Tax |
| 8 | `Property-Tax-Guide-NYC-Class-One.pdf` | Government Guide | New York City Class 1 Tax |
| 9 | `Computer-Repair-Invoice.png` | Out-of-Domain Form | Generic system test |
| 10 | `Property-Maintenance-Invoice-Template.png` | Billing Invoice | Facility Maintenance |

Every chunk carries `source_file` and `doc_type` metadata through the whole pipeline so results can be attributed and filtered.

---

## Chunking Strategy and Reasoning

- **Chunk size:** 600 characters
- **Overlap:** 120 characters (20%)
- **Splitter:** `RecursiveCharacterTextSplitter` (langchain-text-splitters), separator priority `["\n\n", "\n", ". ", " ", ""]`
- **Final chunk count:** 301 chunks across all 10 documents

**Why this fits the documents:** Leases, tax guides, and inspection checklists are dense with specific rules, numbers, and bullet points rather than ordinary prose. 600 characters is large enough to hold a self-contained rule with its surrounding context — verified: the NYC "6% per year / 20% over five years" cap and the HUD "two square feet / 10%" paint threshold each land intact in a single chunk — but small enough to avoid bundling unrelated topics. The recursive splitter prefers paragraph/line boundaries before resorting to mid-sentence cuts, which keeps tabular rows from being severed. The 120-character overlap is a safety net so a figure near a boundary is not chopped in half.

**Extraction preprocessing (added during implementation):** Naive `extract_text()` produced unusable output on several files, so ingestion was hardened before chunking:
- `pdfplumber` with `x_tolerance=1` fixes jammed words (HUD's `"DepartmentofHousing"` → `"Department of Housing"`).
- Pages that fail a quality gate — token-length ratio < 0.5 **or** a run of 6+ consecutive single-character tokens (the signature of rotated/decorative layout, e.g. the NYC cover) — are rasterized with `pypdfium2` and re-read with `easyocr`; OCR output is kept only if it scores higher than the embedded text.
- `.png` images are read with `easyocr` (a single cached `Reader`), requiring no Tesseract system binary.

---

## Sample Chunks

Five labeled chunks (excerpts; each ≤600 chars in storage):

**1. `Property-Tax-Guide-NYC-Class-One.pdf` (Government Guide)**
> "by 6% to arrive at assessed value. There is another factor that affects the calculation of assessed value: State law limits how much the assessed value of a class 1 property … cannot go up more than 6% from the year before or 20% over five years."

**2. `Property-Tax-Guide-Georgia-Citizens.pdf` (Government Guide)**
> "benefits for 2014 is $63,408. The owner must notify the county tax commissioner if for any reason they no longer meet the requirements for this exemption. (O.C.G.A. …)"

**3. `Inspection-Checklist-HUD-HCV.pdf` (Inspection Form)**
> "paint? If not, do deteriorated surfaces exceed two square feet per room and/or is more than 10% of a component? Previous editions are obsolete Page 3 of 8"

**4. `Inspection-Checklist-Property-Management-Smartsheet.pdf` (Inspection Form)**
> "PLUMBING / TURN ON AND OFF ALL FAUCETS … CHECK FOR LEAKS AND DRIPS / WINDOWS / INSPECT ALL GLASS FOR … CHECK FOR AIR LEAKS BY HOLDING A MATCH OR LIGHTER"

**5. `Lease-Agreement-Standard-Residential.pdf` (Lease Contract)**
> "IX. NON-SUFFICIENT FUNDS (NSF CHECKS). If the Tenant pays the Rent with a check that is not honored due to insufficient funds (NSF): (check one) ☐ - There shall …"

---

## Embedding Model

**Model used:** `sentence-transformers/all-MiniLM-L6-v2`, running locally (384-dimensional vectors, CPU). Wired into ChromaDB as a `SentenceTransformerEmbeddingFunction`, so embeddings are produced automatically on `add()` and `query()`.

**Production tradeoff reflection:** Larger cloud-hosted embedding models offer higher recall on domain jargon ("ad valorem", "abatement", "level of assessment") and longer context windows, but introduce per-call cost, network latency, and — critically for legal/financial documents — data-privacy liability from sending sensitive lease and tax data to a third party. For a local-first tool over a small, sensitive corpus, MiniLM is the right default: predictable zero cost, no data leaving the machine, low latency. If deployed for real users with cost no object, I'd weigh a hosted model for better domain recall and multilingual support (non-English tenants) against that privacy cost, and likely keep embeddings local while spending the budget on the generation model instead.

---

## Retrieval Test Results

Top-k = 4, cosine distance (lower = more similar).

**Query A — "In New York City, how much can the assessed value of a Class 1 property increase from one year to the next according to state law caps?"**

| Rank | Distance | Source | Excerpt |
|------|----------|--------|---------|
| 1 | 0.163 | NYC-Class-One | "…State law limits how much the assessed value of a class 1…" |
| 2 | 0.300 | NYC-Class-One | "Market Value — Valuing Your Property…" |
| 3 | 0.300 | NYC-Class-One | "EXAMPLE … Assessed Value if increases were not capped…" |
| 4 | 0.318 | NYC-Class-One | "…by New York State law… go to page 5…" |

*Why relevant:* All four chunks come from the correct NYC guide, and the top hit (distance 0.163, very close) is the exact passage stating the 6%/20% caps. The embedding correctly mapped "assessed value increase per year" + "state law caps" to the capping rule rather than to the surrounding market-value or exemption text.

**Query B — "What is the maximum Social Security benefit amount used for retirement income exclusions under Georgia property tax senior exemptions in 2014?"**

| Rank | Distance | Source | Excerpt |
|------|----------|--------|---------|
| 1 | 0.307 | Georgia-Citizens | "Individuals 65 Years of Age and Older May Claim a $4,000 Exemption…" |
| 2 | 0.416 | Georgia-Citizens | "Income of Applicant and all other persons residing in the home…" |
| 3 | 0.467 | Georgia-Citizens | "benefits for 2014 is $63,408…" |
| 4 | 0.478 | Georgia-Citizens | "retire school bond indebtedness if the income…" |

*Why relevant:* The semantically nearest chunks (1–2) are about senior exemptions and income limits — the right topic — while the chunk holding the literal answer ("$63,408") ranks 3rd at distance 0.467. This shows top-k=4 is correctly sized for this corpus: the exact figure would have been missed at k=2. All hits are correctly confined to the Georgia guide.

**Query C — "What method does the Smartsheet Property Management checklist recommend for checking windows for hidden air leaks?"**

| Rank | Distance | Source | Excerpt |
|------|----------|--------|---------|
| 1 | 0.507 | Smartsheet | "PROPERTY MANAGEMENT INSPECTION CHECKLIST … HEATING & COOLING…" |
| 2 | 0.511 | Smartsheet | "PLUMBING … WINDOWS / INSPECT ALL GLASS … HOLDING A MATCH OR LIGHTER" |
| 3 | 0.577 | HUD-HCV | "Inspection Checklist / Living Room…" |
| 4 | 0.589 | Total-Home | "TOTAL HOME INSPECTION…" |

The answer-bearing chunk ranks 2nd; distances here are higher (~0.5) because the checklist is terse, all-caps, table-style text that embeds less cleanly than prose.

---

## Grounded Generation

Grounding is enforced by three mechanisms in `generator.py`:

**1. System prompt (the primary control):**
> Answer using ONLY the information in the CONTEXT below. Do not use outside or prior knowledge.
> If the answer is not present in the context, reply exactly: "I don't have that information in the loaded documents." Do not guess.
> Cite the source for every factual claim, using the source filename shown in the context, e.g. (Source: Property-Tax-Guide-NYC-Class-One.pdf).
> If the context is unrelated to the question, treat it as no answer and refuse.

**2. Structural — context formatting + attribution:** Retrieved chunks are rendered into a numbered block, each prefixed with `Source: <filename> | Type: <doc_type>`. This gives the model the exact filename to cite and keeps every fact tied to its origin.

**3. Structural — empty-retrieval guard + low temperature:** If retrieval returns nothing, the function returns the refusal string without calling the LLM. `temperature=0.1` keeps the model in extractive, low-creativity mode.

**Source attribution surfaced in output:** the model emits inline `(Source: <filename>.pdf)` citations (see Example Responses), and the Streamlit UI additionally lists every retrieved chunk with its source, type, and distance.

---

## Example Responses

**Example 1 (in-domain, attribution visible):**
> **Q:** In New York City, how much can the assessed value of a Class 1 property increase per year under state law caps?
> **A:** According to state law caps, the assessed value of a Class 1 property in New York City cannot increase by more than 6% from the year before (Source: Property-Tax-Guide-NYC-Class-One.pdf). Additionally, it cannot increase by more than 20% over a five-year period (Source: Property-Tax-Guide-NYC-Class-One.pdf).

**Example 2 (in-domain, attribution visible):**
> **Q:** Per the HUD Housing Choice Voucher checklist, what size threshold determines if deteriorated interior paint fails inspection?
> **A:** According to the HUD Housing Choice Voucher checklist, deteriorated interior paint fails inspection if it exceeds "two square feet per room and/or is more than 10% of a component" (Source: Inspection-Checklist-HUD-HCV.pdf).

**Out-of-scope query (refusal):**
> **Q:** How do I overclock my laptop's GPU for gaming?
> **A:** I don't have that information in the loaded documents.

The retriever still returned 4 chunks for the GPU query, but at high distances (0.857–0.886) and entirely unrelated content; the grounding prompt correctly caused the model to refuse rather than answer from irrelevant context.

---

## Query Interface

A Streamlit app (`app.py`).

**Input:** a single text field ("Ask a question") plus an **Ask** button.
**Output:**
- An **Answer** section with the grounded, source-cited response.
- A collapsible **Retrieved context** panel listing each of the 4 retrieved chunks with its `source_file`, `doc_type`, and cosine distance — so the user can verify the answer against the raw evidence.

**Sample interaction transcript:**
```
[Input]  How much can a NYC Class 1 property's assessed value rise per year?

[Answer] According to state law caps, the assessed value of a Class 1 property
         in New York City cannot increase by more than 6% from the year before
         (Source: Property-Tax-Guide-NYC-Class-One.pdf). Additionally, it cannot
         increase by more than 20% over a five-year period
         (Source: Property-Tax-Guide-NYC-Class-One.pdf).

[Retrieved context]
  1. Property-Tax-Guide-NYC-Class-One.pdf · Government Guide · distance 0.163
  2. Property-Tax-Guide-NYC-Class-One.pdf · Government Guide · distance 0.300
  3. Property-Tax-Guide-NYC-Class-One.pdf · Government Guide · distance 0.300
  4. Property-Tax-Guide-NYC-Class-One.pdf · Government Guide · distance 0.318
```

---

## Evaluation Report

| # | Question | Expected | System Response (summary) | Retrieval | Accuracy |
|---|----------|----------|---------------------------|-----------|----------|
| 1 | Georgia max Social Security benefit for senior exemption (2014) | $63,408 | "$63,408 (Source: Georgia-Citizens.pdf)" | Relevant | **Accurate** |
| 2 | NYC Class 1 assessed-value annual increase cap | 6%/yr, 20%/5yr | "no more than 6% from the year before… 20% over five years (Source: NYC-Class-One.pdf)" | Relevant | **Accurate** |
| 3 | Basic Rental Agreement late-fee penalty + grace day | % of monthly rent; after a set day | "$___ (not to exceed __% of monthly rent); specific day not provided" | Partially relevant | **Partially accurate** |
| 4 | HUD paint-failure size threshold | >2 sq ft/room and/or >10% of component | "exceeds two square feet per room and/or more than 10% of a component (Source: HUD-HCV.pdf)" | Relevant | **Accurate** |
| 5 | Smartsheet window air-leak check method | Hold a match/lighter near window | "CHECK FOR AIR LEAKS BY HOLDING A MATCH OR LIGHTER (Source: Smartsheet.pdf)" | Relevant | **Accurate** |

**Summary:** 4/5 accurate, 1 partially accurate. The out-of-scope GPU query correctly refused.

---

## Failure Case Analysis

**Question that failed:** Q3 — "What is the late fee penalty in the Basic Rental Agreement, and after what day is rent considered late?"

**What the system returned:** "The late fee penalty… is $_____ (not to exceed ___% of the monthly rent)… the specific day after which rent is considered late is not provided in the given context."

**Root cause (two pipeline stages):**
1. **Ingestion / source data.** The Basic Rental Agreement is a *fill-in-the-blank template* — the late-fee percentage and the grace-day are literal blanks (`$_____`, `___%`) in the source PDF. There is no concrete value to extract, so even perfect retrieval cannot surface a number that does not exist in the document.
2. **Retrieval / cross-document confusion.** The top-ranked chunk (distance 0.311) came from the *Standard* lease's NSF-checks clause, not the *Basic* agreement's late-fee clause. The three lease documents share heavy structural jargon ("rent", "check", "fee", "default"), so a late-fee query pulled the wrong lease's adjacent clause ahead of the intended one. The `doc_type` filter cannot disambiguate here because all three are `Lease Contract`.

**What I would change:** (a) Honestly, the most correct behavior is what happened — the template has no value, so reporting the blank is faithful. (b) To improve retrieval precision between near-identical documents, add a finer metadata tag (e.g. `source_variant: Basic/Simple/Standard`) and filter on it, or raise top-k for lease queries so the correct clause has more chance to appear.

---

## Spec Reflection

**One way the spec helped:** The locked architecture parameters (600/120, MiniLM, top-k=4) and the explicit document table in `planning.md` removed a whole class of decisions mid-build. When an AI suggestion proposed shrinking chunks to 400/100, the spec gave a clear basis to reject it — I verified the target answers already fit in 600-char chunks rather than re-tuning on a hunch. The document table also drove the `doc_type` classifier directly.

**One way implementation diverged from the spec, and why:** The spec assumed clean `pdfplumber` text extraction. In practice, the HUD form extracted with jammed words and the NYC guide's decorative cover extracted as vertical single-character garbage. I added an extraction-hardening layer not in the original plan — `x_tolerance` spacing, plus a per-page quality gate that falls back to `pypdfium2` + `easyocr` OCR — because without it, two of the five evaluation questions would have been unanswerable. I also corrected two filenames in the spec table to match the actual files on disk.

---

## AI Usage

**Instance 1 — Ingestion + chunking with extraction hardening.**
- *What I gave the AI:* the `planning.md` Documents and Chunking sections, the real `documents/` listing, and the constraint to stay pure-pip (no Tesseract) at the locked 600/120.
- *What it produced:* `ingest.py` (multi-format extraction + `doc_type` classification) and `chunk.py` (RecursiveCharacterTextSplitter).
- *What I changed/overrode:* After inspecting first-chunk previews, I directed it to add `x_tolerance=1` and a quality-gated OCR fallback — and explicitly **rejected** a later suggestion to drop chunk size to 400/100, because grepping the chunks proved the answers already fit at 600. I kept the blueprint parameters intact.

**Instance 2 — Retrieval + grounded generation.**
- *What I gave the AI:* the requirement that `source_file`/`doc_type` metadata flow end-to-end, top-k=4 cosine, and a strict grounding spec (answer only from context, cite the filename, refuse when absent).
- *What it produced:* `retriever.py` (Chroma cosine collection, `embed_and_store`, `retrieve` with an optional `doc_type` filter), `build_index.py`, `generator.py` (grounding system prompt + attribution + refusal), and the Streamlit `app.py`.
- *What I changed/overrode:* I validated grounding empirically with `run_eval.py` — the out-of-scope GPU query refusing despite 4 returned chunks confirmed the prompt-level grounding worked, so I did not add a separate distance threshold. The honest "partially accurate" Q3 result was kept rather than tuned away.
