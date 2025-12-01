# app/nodes/sanitizer.py

from typing import Dict, Any, List
import re

def clean_page_text(text: str) -> str:
    """Clean a single page's text."""
    if not text:
        return ""
    
    # Normalize whitespace
    text = text.replace("\xa0", " ")   # non-breaking spaces
    text = re.sub(r"\s+", " ", text)   # collapse multiple spaces
    
    # Remove weird control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    
    # Strip ASCII junk/misread OCR artifacts
    text = re.sub(r"[•■□◆◇◦●]", "", text)
    
    # Trim edges
    return text.strip()


def data_cleaner(page_texts: List[str], doc_id: str = None) -> Dict[str, Any]:
    """
    Takes list of raw extracted page texts and returns:
      - cleaned_page_texts
      - cleaned_full_text
      - total characters removed (optional metric)
    """

    cleaned_pages = []
    total_removed = 0

    for page in page_texts:
        before = len(page or "")
        cleaned = clean_page_text(page or "")
        after = len(cleaned)
        total_removed += (before - after)
        cleaned_pages.append(cleaned)

    cleaned_full_text = "\n\n".join([p for p in cleaned_pages if p])

    return {
        "doc_id": doc_id,
        "cleaned_page_texts": cleaned_pages,
        "cleaned_full_text": cleaned_full_text,
        "pages": len(cleaned_pages),
        "chars_removed": total_removed
    }
