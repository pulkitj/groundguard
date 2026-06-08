"""Tests for optional loader helpers."""
import sys
import pytest
from unittest.mock import patch

from groundguard.loaders.helpers import pdf_to_text, docx_to_text


def test_pdf_to_text_missing_dep_raises_import_error():
    """pdf_to_text raises ImportError with install hint when pypdf is absent."""
    with patch.dict(sys.modules, {'pypdf': None}):
        with pytest.raises(ImportError, match="pip install agentic-verifier\\[loaders\\]"):
            pdf_to_text("any_path.pdf")


def test_docx_to_text_missing_dep_raises_import_error():
    """docx_to_text raises ImportError with install hint when python-docx is absent."""
    with patch.dict(sys.modules, {'docx': None}):
        with pytest.raises(ImportError, match="pip install agentic-verifier\\[loaders\\]"):
            docx_to_text("any_path.docx")
