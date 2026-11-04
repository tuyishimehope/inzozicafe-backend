from collections import Counter

from app.services.chunking import chunk_all


def test_counts_per_file():
    counts = Counter(chunk.file_name for chunk in chunk_all())
    assert counts == {
        "menu.xml": 7,
        "branches.xml": 4,
        "delivery.xml": 4,
        "leave-policy.xml": 8,
        "refund-policy.xml": 7,
    }


def test_every_chunk_can_be_cited():
    chunks = chunk_all()
    assert len(chunks) == 30
    assert all(chunk.file_name and chunk.element_id for chunk in chunks)
    assert len({(c.file_name, c.element_id) for c in chunks}) == len(chunks)


def test_no_missing_values_leak_into_text():
    for chunk in chunk_all():
        assert "None" not in chunk.text, chunk
        assert "<Element" not in chunk.text, chunk
        assert "  " not in chunk.text, chunk


def by_id(element_id):
    return next(c for c in chunk_all() if c.element_id == element_id)


def test_facts_the_questions_need_are_in_the_text():
    assert "vegan: no" in by_id("D-02").text  # Masala Chai vegan?
    assert "peanuts" in by_id("F-04").text  # peanut items
    assert "Sunday from 09:00 to 18:00" in by_id("B-KIM").text  # Kimihurura Sunday close
    assert "closed Saturday and Sunday" in by_id("B-KAC").text  # Kacyiru on Saturday
    assert "18 working days" in by_id("L-1.2").text  # leave after 2 years
    assert "fee: 2500 RWF" in by_id("DL-B").text  # Nyamirambo delivery fee
    assert "Gift cards cannot be refunded" in by_id("RF-3").text
    assert "RF-3" in by_id("RF-3").text  # the id is in the text for keyword search later