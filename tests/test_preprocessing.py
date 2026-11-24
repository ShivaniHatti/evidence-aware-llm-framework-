from src.preprocessing.processor import InputProcessor


def test_empty_input():
    p = InputProcessor()
    r = p.process("")
    assert r.is_valid is False
    assert "empty" in (r.error_message or "").lower()


def test_none_input():
    p = InputProcessor()
    r = p.process(None)
    assert r.is_valid is False


def test_whitespace_only():
    p = InputProcessor()
    r = p.process("   \n\t  ")
    assert r.is_valid is False


def test_valid_input_preserves_original():
    p = InputProcessor()
    raw = "  Hello world. Verify your account.  "
    r = p.process(raw)
    assert r.is_valid is True
    assert r.original_text == raw
    assert "Verify your account" in r.normalized_text
    assert len(r.sentence_spans) >= 1
