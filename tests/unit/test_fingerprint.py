import hashlib

import pytest

from app.ingestion.fingerprint import fingerprint, normalize_text_for_hash


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("\ufeffAnna", "Anna", id="bom"),
        pytest.param("\uff21\uff4e\uff4e\uff41 \ufb01sh", "Anna fish", id="nfkc"),
        pytest.param("Anna\u00a0Petrova", "Anna Petrova", id="nfkc-nbsp"),
        pytest.param("one\r\ntwo\rthree", "one\ntwo\nthree", id="line-endings"),
        pytest.param("one  \t\ntwo ", "one\ntwo", id="trailing-whitespace"),
        pytest.param("Anna \t  met\t\tMark", "Anna met Mark", id="inner-runs"),
        pytest.param("one\n\n\n\n\ntwo", "one\n\ntwo", id="blank-line-runs"),
        pytest.param("one\n  \t\n \n\ntwo", "one\n\ntwo", id="whitespace-only-lines"),
        pytest.param("one\n\ntwo", "one\n\ntwo", id="paragraph-kept"),
        pytest.param("\n\n  Anna\n\n", "Anna", id="outer-strip"),
    ],
)
def test_normalize_text_for_hash(raw: str, expected: str) -> None:
    assert normalize_text_for_hash(raw) == expected


def test_normalize_keeps_case_and_punctuation() -> None:
    assert normalize_text_for_hash("Anna, MARK!") == "Anna, MARK!"


def test_fingerprint_is_sha256_hex() -> None:
    assert fingerprint(b"Anna met Mark.") == hashlib.sha256(b"Anna met Mark.").hexdigest()


def test_noisy_variants_normalize_identically() -> None:
    clean = "# Title\n\nAnna met Mark."

    assert normalize_text_for_hash("\ufeff# Title  \r\n\r\n\r\nAnna  met Mark.\r\n") == clean
