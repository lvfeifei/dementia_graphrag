"""
Step 10: ICS calendar reminder generator.

This tool reads Step 05 retrieval result and generates calendar reminders.

Input by default:
- outputs/retrieval_debug/last_retrieval_result.json

Outputs:
- outputs/calendars/last_care_reminders.ics
- outputs/calendars/dementia_care_reminders_<timestamp>.ics

This script does NOT modify Neo4j.
It only reads local retrieval JSON and writes .ics files.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RETRIEVAL_DEBUG_DIR = PROJECT_ROOT / "outputs" / "retrieval_debug"
CALENDAR_DIR = PROJECT_ROOT / "outputs" / "calendars"

DEFAULT_RETRIEVAL_RESULT_PATH = RETRIEVAL_DEBUG_DIR / "last_retrieval_result.json"
LAST_ICS_PATH = CALENDAR_DIR / "last_care_reminders.ics"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_dirs() -> None:
    CALENDAR_DIR.mkdir(parents=True, exist_ok=True)


def load_retrieval_result(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Retrieval result not found: {path}\n"
            "Please run Step 05 or Step 08 first."
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Retrieval result should be a JSON object: {path}")

    return data


def escape_ics_text(text: Any) -> str:
    text = clean_text(text)
    text = text.replace("\\", "\\\\")
    text = text.replace(",", "\\,")
    text = text.replace(";", "\\;")
    text = text.replace("\n", "\\n")
    return text


def fold_ics_line(line: str, limit: int = 73) -> str:
    """
    Fold long ICS lines.
    Strictly speaking, folding should be byte-based.
    For Chinese local calendar import, this simple char-based folding is usually acceptable.
    """
    if len(line) <= limit:
        return line

    parts = []
    while len(line) > limit:
        parts.append(line[:limit])
        line = " " + line[limit:]
    parts.append(line)
    return "\r\n".join(parts)


def format_dt(dt: datetime) -> str:
    """
    Floating local time.
    Apple Calendar and many clients can import it as local time.
    """
    return dt.strftime("%Y%m%dT%H%M%S")


def build_event(
    summary: str,
    description: str,
    begin: datetime,
    minutes: int = 10,
    rrule: Optional[str] = None,
) -> str:
    end = begin + timedelta(minutes=minutes)

    lines = [
        "BEGIN:VEVENT",
        f"UID:{uuid.uuid4()}@dementia-care-graphrag",
        f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        f"DTSTART:{format_dt(begin)}",
        f"DTEND:{format_dt(end)}",
        f"SUMMARY:{escape_ics_text(summary)}",
        f"DESCRIPTION:{escape_ics_text(description)}",
    ]

    if rrule:
        lines.append(f"RRULE:{rrule}")

    lines.append("END:VEVENT")

    return "\r\n".join(fold_ics_line(line) for line in lines)


def get_final_context(retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
    context = retrieval_result.get("final_context") or {}
    if not isinstance(context, dict):
        return {}
    return context


def collect_advice_items(retrieval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    context = get_final_context(retrieval_result)

    items: List[Dict[str, Any]] = []

    for group_name in ["problem_advice", "activity_advice", "environment_advice"]:
        for item in context.get(group_name) or []:
            if not isinstance(item, dict):
                continue
            new_item = dict(item)
            new_item["group_name"] = group_name
            items.append(new_item)

    return items


def item_text(item: Dict[str, Any]) -> str:
    parts = [
        item.get("doc_id"),
        item.get("advice_id"),
        item.get("advice_name"),
        item.get("advice_text"),
        " ".join(item.get("related_problem_ids") or []),
        " ".join(item.get("related_problem_names") or []),
        item.get("source_title"),
        item.get("heading_path"),
        item.get("evidence_text"),
    ]
    return "\n".join(clean_text(x) for x in parts if clean_text(x))


def make_description(item: Dict[str, Any]) -> str:
    doc_id = clean_text(item.get("doc_id"))
    advice_text = clean_text(item.get("advice_text"))
    source_title = clean_text(item.get("source_title"))
    heading_path = clean_text(item.get("heading_path"))

    lines = []
    if doc_id:
        lines.append(f"知识条目：{doc_id}")
    if advice_text:
        lines.append(f"照护建议：{advice_text}")
    if source_title:
        lines.append(f"来源：{source_title}")
    if heading_path:
        lines.append(f"路径：{heading_path}")

    return "\n".join(lines)


def add_event_once(
    events: List[Dict[str, Any]],
    event_key_set: set,
    key: str,
    summary: str,
    description: str,
    begin: datetime,
    minutes: int = 10,
    rrule: Optional[str] = None,
    source_doc_id: str = "",
) -> None:
    if key in event_key_set:
        return

    event_key_set.add(key)

    events.append(
        {
            "key": key,
            "summary": summary,
            "description": description,
            "begin": begin,
            "minutes": minutes,
            "rrule": rrule,
            "source_doc_id": source_doc_id,
        }
    )


def infer_calendar_events(
    retrieval_result: Dict[str, Any],
    days: int = 7,
    start_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Infer practical reminders from structured advice.

    Current rule-based v0.1:
    - natural light / 光照 -> daily morning
    - time orientation / 日期星期季节 -> daily morning and evening cue
    - aromatherapy / 精油 / 芳香 -> nightly
    - sleep environment / 睡眠环境 -> one-time evening setup
    - environment safety / 标识/杂物/危险物品/门磁/摄像头 -> one-time safety check
    - walking route / 空间定向/熟悉路线 -> daily afternoon training
    - drinking before sleep / 睡前饮水 -> nightly water control
    - urination before sleep / 睡前排尿 -> nightly
    """
    items = collect_advice_items(retrieval_result)

    if start_date is None:
        tomorrow = datetime.now() + timedelta(days=1)
    else:
        tomorrow = start_date

    tomorrow = tomorrow.replace(second=0, microsecond=0)

    events: List[Dict[str, Any]] = []
    keys = set()

    environment_texts: List[str] = []

    for item in items:
        text = item_text(item)
        doc_id = clean_text(item.get("doc_id"))
        description = make_description(item)

        if not text:
            continue

        # 1. Natural light.
        if any(k in text for k in ["自然光照", "光照", "明亮环境", "白天保证充足光照"]):
            begin = tomorrow.replace(hour=9, minute=0)
            add_event_once(
                events,
                keys,
                key="daily_morning_light",
                summary="照护提醒：上午接受自然光照",
                description=description,
                begin=begin,
                minutes=120,
                rrule=f"FREQ=DAILY;COUNT={days}",
                source_doc_id=doc_id,
            )

        # 2. Time orientation.
        if any(k in text for k in ["日历", "星期", "日期", "季节", "时间定向", "定向力"]):
            begin = tomorrow.replace(hour=8, minute=30)
            add_event_once(
                events,
                keys,
                key="daily_morning_orientation",
                summary="照护提醒：早晨时间定向提示",
                description=description,
                begin=begin,
                minutes=10,
                rrule=f"FREQ=DAILY;COUNT={days}",
                source_doc_id=doc_id,
            )

            begin2 = tomorrow.replace(hour=20, minute=30)
            add_event_once(
                events,
                keys,
                key="daily_evening_orientation",
                summary="照护提醒：夜间睡前时间提示",
                description=description,
                begin=begin2,
                minutes=10,
                rrule=f"FREQ=DAILY;COUNT={days}",
                source_doc_id=doc_id,
            )

        # 3. Aromatherapy.
        if any(k in text for k in ["精油", "芳香", "薰衣草"]):
            begin = tomorrow.replace(hour=20, minute=30)
            add_event_once(
                events,
                keys,
                key="night_aromatherapy",
                summary="照护提醒：晚间放松助眠",
                description=description,
                begin=begin,
                minutes=15,
                rrule=f"FREQ=DAILY;COUNT={days}",
                source_doc_id=doc_id,
            )

        # 4. Sleep environment.
        if any(k in text for k in ["睡眠环境", "卧室", "遮光窗帘", "夜间环境黑暗", "减少夜间噪音", "床铺"]):
            begin = tomorrow.replace(hour=19, minute=30)
            add_event_once(
                events,
                keys,
                key="evening_sleep_environment_check",
                summary="照护任务：检查睡眠环境",
                description=description,
                begin=begin,
                minutes=20,
                rrule=f"FREQ=DAILY;COUNT={days}",
                source_doc_id=doc_id,
            )

        # 5. Sleep water / urination.
        if any(k in text for k in ["睡前2小时", "睡前两小时", "控制饮水", "减少饮水"]):
            begin = tomorrow.replace(hour=20, minute=0)
            add_event_once(
                events,
                keys,
                key="night_water_control",
                summary="照护提醒：睡前控制饮水",
                description=description,
                begin=begin,
                minutes=5,
                rrule=f"FREQ=DAILY;COUNT={days}",
                source_doc_id=doc_id,
            )

        if "睡前排尿" in text or "睡前上厕所" in text:
            begin = tomorrow.replace(hour=21, minute=30)
            add_event_once(
                events,
                keys,
                key="night_toilet_before_sleep",
                summary="照护提醒：睡前排尿",
                description=description,
                begin=begin,
                minutes=5,
                rrule=f"FREQ=DAILY;COUNT={days}",
                source_doc_id=doc_id,
            )

        # 6. Walking / orientation training.
        if any(k in text for k in ["空间定向", "熟悉路线", "多次往返", "方位词", "散步"]):
            begin = tomorrow.replace(hour=15, minute=30)
            add_event_once(
                events,
                keys,
                key="daily_orientation_walk",
                summary="照护训练：熟悉路线与空间定向",
                description=description,
                begin=begin,
                minutes=25,
                rrule=f"FREQ=DAILY;COUNT={days}",
                source_doc_id=doc_id,
            )

        # 7. Environment one-time safety check.
        if (
            item.get("group_name") == "environment_advice"
            or any(k in text for k in ["标识", "杂物", "危险物品", "门磁", "摄像头", "报警", "出口", "镜子", "反光", "易碎"])
        ):
            environment_texts.append(f"{doc_id}：{clean_text(item.get('advice_text'))}")

    if environment_texts:
        begin = tomorrow.replace(hour=10, minute=0)
        env_description = "请根据以下建议检查并调整居家环境：\n" + "\n".join(environment_texts[:8])
        add_event_once(
            events,
            keys,
            key="one_time_environment_safety_check",
            summary="照护任务：检查居家环境风险",
            description=env_description,
            begin=begin,
            minutes=40,
            rrule=None,
            source_doc_id="environment_advice",
        )

    return events


def build_ics_text(events: List[Dict[str, Any]]) -> str:
    event_blocks: List[str] = []

    for event in events:
        event_blocks.append(
            build_event(
                summary=event["summary"],
                description=event["description"],
                begin=event["begin"],
                minutes=event.get("minutes", 10),
                rrule=event.get("rrule"),
            )
        )

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//DementiaCare GraphRAG Agent//CN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        *event_blocks,
        "END:VCALENDAR",
    ]

    return "\r\n".join(lines) + "\r\n"


def generate_calendar_from_retrieval_result(
    retrieval_result_path: Path = DEFAULT_RETRIEVAL_RESULT_PATH,
    days: int = 7,
) -> Dict[str, Any]:
    ensure_dirs()

    retrieval_result = load_retrieval_result(retrieval_result_path)
    events = infer_calendar_events(retrieval_result, days=days)

    if not events:
        raise ValueError(
            "No calendar reminders were inferred from this retrieval result. "
            "Try another question or inspect final_context."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_ics_path = CALENDAR_DIR / f"dementia_care_reminders_{timestamp}.ics"

    ics_text = build_ics_text(events)

    timestamp_ics_path.write_text(ics_text, encoding="utf-8")
    shutil.copy2(timestamp_ics_path, LAST_ICS_PATH)

    event_preview = [
        {
            "summary": e["summary"],
            "begin": e["begin"].isoformat(timespec="minutes"),
            "minutes": e.get("minutes", 10),
            "rrule": e.get("rrule"),
            "source_doc_id": e.get("source_doc_id"),
        }
        for e in events
    ]

    return {
        "timestamp_ics_path": str(timestamp_ics_path),
        "last_ics_path": str(LAST_ICS_PATH),
        "event_count": len(events),
        "events": event_preview,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 10: Generate ICS care reminders.")
    parser.add_argument(
        "--retrieval_result",
        type=str,
        default="",
        help="Path to retrieval result JSON. Default: outputs/retrieval_debug/last_retrieval_result.json",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days for recurring reminders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    retrieval_result_path = Path(args.retrieval_result) if args.retrieval_result else DEFAULT_RETRIEVAL_RESULT_PATH
    if not retrieval_result_path.is_absolute():
        retrieval_result_path = PROJECT_ROOT / retrieval_result_path

    print("=" * 80)
    print("Step 10: ICS calendar reminder generator")
    print("=" * 80)
    print(f"Using retrieval result: {retrieval_result_path}")

    result = generate_calendar_from_retrieval_result(
        retrieval_result_path=retrieval_result_path,
        days=args.days,
    )

    print("\n[PASS] ICS calendar generated.")
    print(f"[OK] Timestamp ICS: {result['timestamp_ics_path']}")
    print(f"[OK] Last ICS: {result['last_ics_path']}")
    print(f"[OK] Event count: {result['event_count']}")

    print("\nEvents:")
    for event in result["events"]:
        print(f"- {event['begin']} | {event['summary']} | {event.get('rrule') or 'one-time'}")


if __name__ == "__main__":
    main()
