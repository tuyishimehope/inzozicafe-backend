"""Turn the five Inzozi XML files into chunks.

One chunk per item, rule, branch or zone. Every chunk keeps:
  - file_name:  which file it came from (for the citation)
  - element_id: the id of the element inside that file (for the citation)
  - text:       a readable sentence that starts with a breadcrumb, e.g.
                "Menu > Drinks > Masala Chai (D-02): price 2000 RWF; ..."

The id is also written into the text on purpose: later, keyword search
can find "RF-3" even though a vector search cannot.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")


@dataclass(frozen=True)
class ChunkData:
    file_name: str
    element_id: str
    text: str


# ---------- small helpers (each does one job) ----------


def _load(path: Path) -> ET.Element:
    """Open an XML file and return its root element."""
    return ET.parse(path).getroot()


def _id(element: ET.Element) -> str:
    """The id attribute. Fail loudly: a chunk without an id cannot be cited."""
    value = element.get("id")
    if not value:
        raise ValueError(f"<{element.tag}> has no id attribute")
    return value


def _text(element: ET.Element, tag: str) -> str:
    """Text of a child tag, trimmed. Empty string if the tag is missing."""
    return (element.findtext(tag) or "").strip()


def _own_text(element: ET.Element) -> str:
    """Text written directly inside the element, trimmed."""
    return (element.text or "").strip()


def _yes_no(value: str) -> str:
    return "yes" if value.strip().lower() == "true" else "no"


def _money(value: str, currency: str) -> str:
    """'1500' -> '1500 RWF', but leave 'not available' alone."""
    return f"{value} {currency}" if value.isdigit() else value


# ---------- one function per file ----------


def chunk_menu(path: Path) -> list[ChunkData]:
    root = _load(path)
    currency = root.get("currency", "")
    chunks = []
    for category in root.findall("category"):
        for item in category.findall("item"):
            item_id = _id(item)
            text = (
                f"Menu > {category.get('name')} > {_text(item, 'name')} ({item_id}): "
                f"price {_money(_text(item, 'price'), currency)}; "
                f"vegan: {_yes_no(_text(item, 'vegan'))}; "
                f"allergens: {_text(item, 'allergens')}. "
                f"{_text(item, 'description')}"
            )
            chunks.append(ChunkData(path.name, item_id, text))
    return chunks


def _hours_slot(slot: ET.Element) -> str:
    """One <open .../> or <closed .../> line as words."""
    days = slot.get("days", "")
    if slot.tag == "closed":
        return f"closed {days}"
    return f"open {days} from {slot.get('from')} to {slot.get('to')}"


def chunk_branches(path: Path) -> list[ChunkData]:
    root = _load(path)
    chunks = []
    for branch in root.findall("branch"):
        branch_id = _id(branch)
        hours = "; ".join(_hours_slot(slot) for slot in branch.findall("hours/*"))
        text = (
            f"Branches > {_text(branch, 'name')} ({branch_id}): "
            f"address {_text(branch, 'address')}; "
            f"phone {_text(branch, 'phone')}; "
            f"hours: {hours}. "
            f"{_text(branch, 'notes')}"
        )
        chunks.append(ChunkData(path.name, branch_id, text))

    # <holidays> is not a <branch>, but it is a fact people ask about, so it is a chunk too.
    for holidays in root.findall("holidays"):
        rules = " ".join(_own_text(rule) for rule in holidays.findall("rule"))
        chunks.append(
            ChunkData(
                path.name,
                _id(holidays),
                f"Branches > Holidays ({_id(holidays)}): {rules}",
            )
        )
    return chunks


def chunk_delivery(path: Path) -> list[ChunkData]:
    root = _load(path)
    currency = root.get("currency", "")
    chunks = []

    for general in root.findall("general"):
        general_id = _id(general)
        hours = general.find("orderHours")
        order_hours = (
            f"{hours.get('from')} to {hours.get('to')}" if hours is not None else ""
        )
        text = (
            f"Delivery > General ({general_id}): "
            f"orders are taken from {order_hours}; "
            f"minimum order {_money(_text(general, 'minimumOrder'), currency)}; "
            f"payment: {_text(general, 'payment')}"
        )
        chunks.append(ChunkData(path.name, general_id, text))

    for zone in root.findall("zone"):
        zone_id = _id(zone)
        text = (
            f"Delivery > {zone.get('name')} ({zone_id}): "
            f"areas: {_text(zone, 'areas')}; "
            f"fee: {_money(_text(zone, 'fee'), currency)}; "
            f"time: {_text(zone, 'time')}. "
            f"{_text(zone, 'freeDelivery')}"
        )
        chunks.append(ChunkData(path.name, zone_id, text))
    return chunks


def chunk_leave_policy(path: Path) -> list[ChunkData]:
    root = _load(path)
    policy = root.get("name", "")
    chunks = []
    for section in root.findall("section"):
        for rule in section.findall("rule"):
            rule_id = _id(rule)
            text = f"{policy} > {section.get('title')} > {rule_id}: {_own_text(rule)}"
            chunks.append(ChunkData(path.name, rule_id, text))
    return chunks


def chunk_refund_policy(path: Path) -> list[ChunkData]:
    root = _load(path)
    policy = root.get("name", "")
    chunks = []
    for rule in root.findall("rule"):
        rule_id = _id(rule)
        chunks.append(
            ChunkData(path.name, rule_id, f"{policy} > {rule_id}: {_own_text(rule)}")
        )
    return chunks


# ---------- put it together ----------

PARSERS = {
    "menu.xml": chunk_menu,
    "branches.xml": chunk_branches,
    "delivery.xml": chunk_delivery,
    "leave-policy.xml": chunk_leave_policy,
    "refund-policy.xml": chunk_refund_policy,
}


def chunk_all(data_dir: Path = DATA_DIR) -> list[ChunkData]:
    chunks: list[ChunkData] = []
    for file_name, parse in PARSERS.items():
        chunks.extend(parse(data_dir / file_name))

    # A citation is (file, id). If two chunks share it, a citation points at two places.
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        key = (chunk.file_name, chunk.element_id)
        if key in seen:
            raise ValueError(f"duplicate chunk {key}")
        seen.add(key)
    return chunks
