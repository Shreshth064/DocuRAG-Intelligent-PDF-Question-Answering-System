import pytest

from rag.environment import describe, missing, require


def test_missing_reports_absent_variables(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)

    assert missing("SOME_KEY") == ["SOME_KEY"]


def test_missing_treats_empty_string_as_absent(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "")

    assert missing("SOME_KEY") == ["SOME_KEY"]


def test_missing_returns_empty_when_all_present(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "value")

    assert missing("SOME_KEY") == []


def test_describe_includes_a_hint_for_known_keys():
    message = describe(["GOOGLE_API_KEY"])

    assert "GOOGLE_API_KEY" in message
    assert "aistudio.google.com" in message


def test_describe_omits_hint_for_unknown_keys():
    message = describe(["MYSTERY_KEY"])

    key_line = next(line for line in message.splitlines() if "MYSTERY_KEY" in line)
    assert key_line.strip() == "- MYSTERY_KEY"  # no parenthesised hint


def test_require_raises_with_a_readable_message_when_absent(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        require("GOOGLE_API_KEY")

    assert "GOOGLE_API_KEY" in str(exc_info.value)


def test_require_passes_silently_when_present(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "value")

    assert require("GOOGLE_API_KEY") is None
