"""Turn the five Inzozi XML files into chunks.

One chunk per item, rule, branch or zone. Every chunk keeps:
  - file_name:  which file it came from (for the citation)
  - element_id: the id of the element inside that file (for the citation)
  - text:       a readable sentence that starts with a breadcrumb, e.g.
                "Menu > Drinks > Masala Chai (D-02): price 2000 RWF; ..."

The id is also written into the text on purpose: keyword search can find
"RF-3" even though a vector search cannot.

Each file has one parser. A parser gets the file's root element and yields
(element_id, text) pairs; `chunk_file` turns those into ChunkData.
"""

import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")

Pairs = Iterator[tuple[str, str]]  # (element_id, text)


@dataclass(frozen=True)
class ChunkData:
    file_name: str
    element_id: str
    text: str


# ---------- helpers ----------


def _id(element: ET.Element) -> str:
    """The id attribute. Fail loudly: a chunk without an id cannot be cited."""
    value = element.get("id")
    if not value:
        raise ValueError(f"<{element.tag}> has no id attribute")
    return value


def _text(element: ET.Element, tag: str | None = None) -> str:
    """Trimmed text of a child tag, or of the element itself when no tag is given."""
    value = element.text if tag is None else element.findtext(tag)
    return (value or "").strip()


def _yes_no(value: str) -> str:
    return "yes" if value.strip().lower() == "true" else "no"


def _money(value: str, currency: str) -> str:
    """'1500' -> '1500 RWF', but leave 'not available' alone."""
    return f"{value} {currency}" if value.isdigit() else value


def _hours(slot: ET.Element) -> str:
    """One <open .../> or <closed .../> line as words."""
    days = slot.get("days", "")
    if slot.tag == "closed":
        return f"closed {days}"
    return f"open {days} from {slot.get('from')} to {slot.get('to')}"


# ---------- one parser per file ----------


def parse_menu(root: ET.Element) -> Pairs:
    currency = root.get("currency", "")
    for category in root.findall("category"):
        for item in category.findall("item"):
            item_id = _id(item)
            yield item_id, (
                f"Menu > {category.get('name')} > {_text(item, 'name')} ({item_id}): "
                f"price {_money(_text(item, 'price'), currency)}; "
                f"vegan: {_yes_no(_text(item, 'vegan'))}; "
                f"allergens: {_text(item, 'allergens')}. "
                f"{_text(item, 'description')}"
            )


def parse_branches(root: ET.Element) -> Pairs:
    for branch in root.findall("branch"):
        branch_id = _id(branch)
        hours = "; ".join(_hours(slot) for slot in branch.findall("hours/*"))
        yield branch_id, (
            f"Branches > {_text(branch, 'name')} ({branch_id}): "
            f"address {_text(branch, 'address')}; "
            f"phone {_text(branch, 'phone')}; "
            f"hours: {hours}. "
            f"{_text(branch, 'notes')}"
        )

    # <holidays> is not a <branch>, but people ask about it, so it is a chunk too.
    for holidays in root.findall("holidays"):
        holidays_id = _id(holidays)
        rules = " ".join(_text(rule) for rule in holidays.findall("rule"))
        yield holidays_id, f"Branches > Holidays ({holidays_id}): {rules}"


def parse_delivery(root: ET.Element) -> Pairs:
    currency = root.get("currency", "")
    for general in root.findall("general"):
        general_id = _id(general)
        hours = general.find("orderHours")
        order_hours = "" if hours is None else f"{hours.get('from')} to {hours.get('to')}"
        yield general_id, (
            f"Delivery > General ({general_id}): "
            f"orders are taken from {order_hours}; "
            f"minimum order {_money(_text(general, 'minimumOrder'), currency)}; "
            f"payment: {_text(general, 'payment')}"
        )

    for zone in root.findall("zone"):
        zone_id = _id(zone)
        yield zone_id, (
            f"Delivery > {zone.get('name')} ({zone_id}): "
            f"areas: {_text(zone, 'areas')}; "
            f"fee: {_money(_text(zone, 'fee'), currency)}; "
            f"time: {_text(zone, 'time')}. "
            f"{_text(zone, 'freeDelivery')}"
        )


def parse_leave_policy(root: ET.Element) -> Pairs:
    policy = root.get("name", "")
    for section in root.findall("section"):
        for rule in section.findall("rule"):
            rule_id = _id(rule)
            yield rule_id, f"{policy} > {section.get('title')} > {rule_id}: {_text(rule)}"


def parse_refund_policy(root: ET.Element) -> Pairs:
    policy = root.get("name", "")
    for rule in root.findall("rule"):
        rule_id = _id(rule)
        yield rule_id, f"{policy} > {rule_id}: {_text(rule)}"


# ---------- put it together ----------

PARSERS: dict[str, Callable[[ET.Element], Pairs]] = {
    "menu.xml": parse_menu,
    "branches.xml": parse_branches,
    "delivery.xml": parse_delivery,
    "leave-policy.xml": parse_leave_policy,
    "refund-policy.xml": parse_refund_policy,
}


def chunk_file(path: Path) -> list[ChunkData]:
    """Parse one known file into chunks. A citation is (file, id), so ids must be unique."""
    parse = PARSERS[path.name]
    chunks = [
        ChunkData(path.name, element_id, text)
        for element_id, text in parse(ET.parse(path).getroot())
    ]
    ids = [c.element_id for c in chunks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path.name}: duplicate ids {duplicates}")
    return chunks


def chunk_all(data_dir: Path = DATA_DIR) -> list[ChunkData]:
    return [chunk for name in PARSERS for chunk in chunk_file(data_dir / name)]
