from research_lab.adapters.crossref import CrossrefAdapter


def test_crossref_record_without_doi_uses_url_as_stable_identifier():
    record = CrossrefAdapter._parse(
        CrossrefAdapter,
        {
            "URL": "https://example.org/work/123",
            "title": ["A DOI-less work"],
            "author": [],
            "published": {"date-parts": [[2026, 9, 23]]},
        },
    )
    assert record.external_id == "https://example.org/work/123"
    assert record.doi is None
    assert str(record.source_url) == "https://example.org/work/123"


def test_crossref_record_with_doi_preserves_normalized_doi():
    record = CrossrefAdapter._parse(
        CrossrefAdapter,
        {
            "DOI": "10.1234/ABC.1",
            "title": ["A work"],
            "author": [],
            "published": {"date-parts": [[2026, 9, 23]]},
        },
    )
    assert record.external_id == "10.1234/ABC.1"
    assert record.doi == "10.1234/abc.1"
