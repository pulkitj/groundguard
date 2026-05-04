"""Tests for optional loader helpers — @pytest.mark.loaders."""
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.loaders
def test_pdf_to_text_missing_dep_raises_import_error():
    """pdf_to_text raises ImportError with install hint when pypdf is absent."""
    import sys
    with patch.dict(sys.modules, {'pypdf': None}):
        from importlib import reload
        import groundguard.loaders.helpers as helpers_mod
        reload(helpers_mod)
        with pytest.raises(ImportError, match="pip install agentic-verifier\\[loaders\\]"):
            helpers_mod.pdf_to_text("any_path.pdf")


@pytest.mark.loaders
def test_docx_to_text_missing_dep_raises_import_error():
    """docx_to_text raises ImportError with install hint when python-docx is absent."""
    import sys
    with patch.dict(sys.modules, {'docx': None}):
        from importlib import reload
        import groundguard.loaders.helpers as helpers_mod
        reload(helpers_mod)
        with pytest.raises(ImportError, match="pip install agentic-verifier\\[loaders\\]"):
            helpers_mod.docx_to_text("any_path.docx")
