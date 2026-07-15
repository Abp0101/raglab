from raglab.ingestion.metadata import detect_section_headings, normalize_page_text


def test_normalization_repairs_hyphenation_and_whitespace() -> None:
    text = "Wearable   rehabilita-\n tion\r\n\r\nuses\tIMUs.\n\n\n\nResults"

    normalized = normalize_page_text(text)

    assert normalized == "Wearable rehabilitation\n\nuses IMUs.\n\nResults"


def test_heading_detection_combines_font_and_text_signals() -> None:
    text = "Abstract\nBody text.\n\n2. Methods\nMore body text.\n\nRESULTS\nEvidence."

    headings = detect_section_headings(text, ["Abstract"])

    assert [heading.text for heading in headings] == ["Abstract", "2. Methods", "RESULTS"]
    assert [heading.start for heading in headings] == sorted(heading.start for heading in headings)
