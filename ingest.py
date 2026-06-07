import os

from config import DOCS_PATH


# ---------------------------------------------------------------------------
# Document-type classification
# ---------------------------------------------------------------------------
# Substring rules, checked in order. Substrings (not exact filenames) so the
# classifier is robust to the suffix differences between the planning.md table
# and the actual files on disk (e.g. "...-Smartsheet.pdf", "...-Inspection.pdf").
DOC_TYPE_RULES = [
    ("lease-agreement", "Lease Contract"),
    ("inspection-checklist", "Inspection Form"),
    ("property-tax-guide", "Government Guide"),
    ("invoice", "Invoice"),
]


def classify_doc_type(filename):
    """Map a filename to its document type using DOC_TYPE_RULES."""
    low = filename.lower()
    for key, doc_type in DOC_TYPE_RULES:
        if key in low:
            return doc_type
    return "Unknown"


# ---------------------------------------------------------------------------
# Per-format text extraction
# ---------------------------------------------------------------------------
def _extract_txt(filepath):
    """Read a plain-text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# A page is OCR'd when its embedded-text quality falls below this. Quality =
# fraction of whitespace-split tokens that are 2+ chars. Decorative/rotated
# layouts (e.g. vertically stacked headings) extract as a pile of single
# characters and score near 0; clean prose scores well above 0.5.
_PAGE_QUALITY_THRESHOLD = 0.5


def _text_quality(text):
    """Fraction of tokens that are 2+ chars. 0.0 for empty text."""
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if len(t) >= 2) / len(tokens)


def _has_vertical_text(text, run=6):
    """
    True if the text contains a run of `run`+ consecutive single-char tokens —
    the signature of vertically stacked / decorative layout (e.g. "c l a s s
    o n e") that survives even when the page is mostly clean prose, so the
    overall quality ratio alone wouldn't trip the OCR fallback.
    """
    streak = 0
    for token in text.split():
        if len(token) == 1:
            streak += 1
            if streak >= run:
                return True
        else:
            streak = 0
    return False


def _ocr_pdf_page(pdfium_doc, page_index, scale=2.0):
    """Rasterize one PDF page and OCR it with easyocr."""
    import numpy as np

    bitmap = pdfium_doc[page_index].render(scale=scale)
    reader = _get_ocr_reader()
    return "\n".join(reader.readtext(np.asarray(bitmap.to_numpy()), detail=0))


def _extract_pdf(filepath):
    """
    Extract text from a PDF, page by page, joined on blank lines.

    Two hardening passes over plain pdfplumber:
      - x_tolerance=1 inserts spaces at smaller glyph gaps, fixing jammed
        words like "DepartmentofHousing" -> "Department of Housing".
      - Pages whose embedded text scores below _PAGE_QUALITY_THRESHOLD (e.g.
        rotated/decorative layouts that extract as single-char garbage) are
        re-read by rasterizing the page and running easyocr on it.
    """
    import pdfplumber

    pieces = []
    pdfium_doc = None  # opened lazily, only if a page needs OCR fallback
    try:
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=1) or ""

                if _text_quality(text) < _PAGE_QUALITY_THRESHOLD or _has_vertical_text(text):
                    if pdfium_doc is None:
                        import pypdfium2 as pdfium

                        pdfium_doc = pdfium.PdfDocument(filepath)
                    ocr_text = _ocr_pdf_page(pdfium_doc, i)
                    # Keep OCR only if it actually beats the embedded text.
                    if _text_quality(ocr_text) > _text_quality(text):
                        text = ocr_text

                if text.strip():
                    pieces.append(text)
    finally:
        if pdfium_doc is not None:
            pdfium_doc.close()

    return "\n\n".join(pieces)


# easyocr.Reader loads a model into memory — build it once and reuse for every
# image rather than rebuilding it per file (the model load is the slow part).
_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(["en"])  # downloads model on first use
    return _ocr_reader


def _extract_image(filepath):
    """OCR text out of an image with easyocr. Pure-pip, no system binary."""
    reader = _get_ocr_reader()
    return "\n".join(reader.readtext(filepath, detail=0))


# Map file extension -> extractor. Add new formats here.
_EXTRACTORS = {
    ".txt": _extract_txt,
    ".pdf": _extract_pdf,
    ".png": _extract_image,
    ".jpg": _extract_image,
    ".jpeg": _extract_image,
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_documents():
    """
    Load every supported document (.txt, .pdf, .png/.jpg) from DOCS_PATH.

    Returns a list of dicts, each carrying the raw text plus the source
    metadata that must travel with it through the rest of the pipeline:
      - "text"        : extracted document text (str)
      - "source_file" : original filename, e.g. "Lease-Agreement-Simple-Form.pdf"
      - "doc_type"    : classified type, e.g. "Lease Contract"
    """
    documents = []
    for filename in sorted(os.listdir(DOCS_PATH)):
        ext = os.path.splitext(filename)[1].lower()
        extractor = _EXTRACTORS.get(ext)
        if extractor is None:
            continue

        filepath = os.path.join(DOCS_PATH, filename)
        try:
            text = extractor(filepath)
        except Exception as e:
            print(f"⚠️  Skipped {filename}: {type(e).__name__}: {e}")
            continue

        # PDFs and OCR can silently yield nothing (scanned PDF, blank image).
        if not text or not text.strip():
            print(f"⚠️  No text extracted from {filename} — skipping.")
            continue

        documents.append({
            "text": text,
            "source_file": filename,
            "doc_type": classify_doc_type(filename),
        })

    summary = [(d["source_file"], d["doc_type"]) for d in documents]
    print(f"Loaded {len(documents)} document(s):")
    for source_file, doc_type in summary:
        print(f"  - {source_file}  [{doc_type}]")
    return documents


if __name__ == "__main__":
    load_documents()
