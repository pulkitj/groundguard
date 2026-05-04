"""RED tests for structured loaders (T-97).

load_docx and load_xlsx live in groundguard/loaders/structured.py (not yet
implemented). load_legal_docx lives in groundguard/loaders/legal.py (not yet
implemented). All tests must FAIL until the respective implementation exists.

Run the non-legal subset once Phase 30a is implemented:
    pytest tests/test_structured_loader.py -m loaders -k "not legal_docx" -x -q
"""
from __future__ import annotations

import pathlib

import pytest

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
docx_path = FIXTURES_DIR / "sample.docx"
xlsx_path = FIXTURES_DIR / "sample.xlsx"
legal_docx_path = FIXTURES_DIR / "legal_sample.docx"


@pytest.mark.loaders
def test_load_docx_split_by_heading(loader_fixtures):
    from groundguard.loaders.structured import load_docx

    sources = load_docx(str(docx_path), split_by="heading")
    assert len(sources) >= 2
    for s in sources:
        assert s.source_id
        assert s.content


@pytest.mark.loaders
def test_load_docx_split_by_paragraph(loader_fixtures):
    from groundguard.loaders.structured import load_docx

    sources = load_docx(str(docx_path), split_by="paragraph")
    assert len(sources) >= 1


@pytest.mark.loaders
def test_load_docx_source_type_passes_through(loader_fixtures):
    from groundguard.loaders.structured import load_docx

    sources = load_docx(str(docx_path), source_type="legal_clause")
    assert all(s.source_type == "legal_clause" for s in sources)


@pytest.mark.loaders
def test_load_xlsx_source_type_financial(loader_fixtures):
    from groundguard.loaders.structured import load_xlsx

    sources = load_xlsx(str(xlsx_path), source_type="financial_table_row")
    assert all(s.source_type == "financial_table_row" for s in sources)
    assert any("Revenue" in s.content or ":" in s.content for s in sources)


@pytest.mark.loaders
def test_load_xlsx_reads_all_sheets(tmp_path):
    """Multi-sheet workbook — all sheets are read, not just the active one."""
    import openpyxl
    from groundguard.loaders.structured import load_xlsx

    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1.append(["Name", "Value"])
    ws1.append(["Alpha", "100"])

    ws2 = wb.create_sheet("Sheet2")
    ws2.append(["Name", "Value"])
    ws2.append(["Beta", "200"])

    path = tmp_path / "multi.xlsx"
    wb.save(str(path))

    sources = load_xlsx(str(path))
    assert len(sources) == 2
    contents = [s.content for s in sources]
    assert any("Alpha" in c for c in contents)
    assert any("Beta" in c for c in contents)
    assert any("Sheet1" in s.source_id for s in sources)
    assert any("Sheet2" in s.source_id for s in sources)


@pytest.mark.loaders
def test_load_legal_docx_marks_definitions(loader_fixtures):
    from groundguard.loaders.legal import load_legal_docx

    sources = load_legal_docx(str(legal_docx_path))
    def_sources = [s for s in sources if s.source_type == "legal_definition"]
    assert len(def_sources) >= 1
