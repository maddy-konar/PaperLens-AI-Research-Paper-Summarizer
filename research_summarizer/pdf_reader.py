"""Extract selectable text from PDF pages; scanned PDFs need OCR elsewhere."""
from __future__ import annotations
import re
from dataclasses import dataclass
import fitz

MAX_PAGES = 80
MAX_BYTES = 20 * 1024 * 1024
MAX_CHARS = 450_000

@dataclass(frozen=True)
class PageText:
    page: int
    text: str


def clean_page_text(text: str) -> str:
    """Conservatively repair line wrapping and hyphenated line breaks."""
    text = text.replace('\u00ad', '').replace('\x00', '')
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)
    # Preserve clear standalone section headings before merging wrapped body lines.
    headings = ('abstract', 'introduction', 'background', 'related work', 'literature review',
                'methods', 'methodology', 'results', 'discussion', 'conclusion', 'conclusions',
                'future work', 'references', 'bibliography')
    lines = text.splitlines()
    text = '\n'.join(('\n'+line.strip()+'\n') if line.strip().lower() in headings else line
                     for line in lines)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def extract_pdf(pdf_bytes: bytes) -> list[PageText]:
    if not pdf_bytes or len(pdf_bytes) > MAX_BYTES:
        raise ValueError('Upload a non-empty PDF smaller than 20 MB.')
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    except Exception as exc:
        raise ValueError('The PDF could not be opened. Try another readable PDF.') from exc
    with doc:
        if doc.is_encrypted:
            raise ValueError('Password-protected PDFs are not supported.')
        if doc.page_count > MAX_PAGES:
            raise ValueError(f'PDF exceeds the {MAX_PAGES}-page limit.')
        pages: list[PageText] = []
        char_count = 0
        for index in range(doc.page_count):
            # "sort" is a best-effort reading order, not a layout/column guarantee.
            page = doc.load_page(index)
            # Ignore marginal header/footer text when it is safely outside body area.
            # This is conservative, best-effort filtering for research PDF layouts.
            blocks = page.get_text('blocks', sort=True)
            raw = '\n'.join(str(b[4]) for b in blocks
                            if len(b) >= 7 and b[6] == 0
                            and b[1] >= 26 and b[3] <= page.rect.height - 44)
            text = clean_page_text(raw)
            char_count += len(text)
            if char_count > MAX_CHARS:
                raise ValueError('Extracted text exceeds the 450,000-character limit.')
            if text:
                pages.append(PageText(index + 1, text))
    if sum(len(p.text.strip()) for p in pages) < 120:
        raise ValueError('Little selectable text was found. A scanned PDF needs OCR before upload.')
    return pages


def pdf_to_sentences(pdf_bytes: bytes):
    """Convenience adapter; use pipeline.make_sentences to retain page numbers."""
    from .pipeline import make_sentences
    return make_sentences(extract_pdf(pdf_bytes))
