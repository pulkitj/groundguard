"""Optional PDF/DOCX loader helpers."""
from __future__ import annotations


def pdf_to_text(path: str) -> str:
    """
    Extract text from a PDF file.

    Requires: pip install agentic-verifier[loaders]

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text as a string.

    Raises:
        ImportError: If pypdf is not installed.
    """
    try:
        import pypdf
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF loading. "
            "Install it with: pip install agentic-verifier[loaders]"
        )
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def docx_to_text(path: str) -> str:
    """
    Extract text from a DOCX file.

    Requires: pip install agentic-verifier[loaders]

    Args:
        path: Path to the DOCX file.

    Returns:
        Extracted text as a string.

    Raises:
        ImportError: If python-docx is not installed.
    """
    try:
        import docx
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX loading. "
            "Install it with: pip install agentic-verifier[loaders]"
        )
    doc = docx.Document(path)
    return "\n".join(para.text for para in doc.paragraphs)
