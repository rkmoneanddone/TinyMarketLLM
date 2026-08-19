from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from analyze_chart import analyze_chart
from validate_record import validate_analysis


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp",
    ".bmp", ".gif", ".tif", ".tiff"
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCOMING = PROJECT_ROOT / "charts" / "incoming"
PROCESSED = PROJECT_ROOT / "charts" / "processed"
FAILED = PROJECT_ROOT / "charts" / "failed"
REVIEW = PROJECT_ROOT / "charts" / "review"

RAW_RECORDS = PROJECT_ROOT / "data" / "raw" / "chart_records.jsonl"
LEDGER = PROJECT_ROOT / "data" / "metadata" / "processed_files.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ledger() -> dict:
    if not LEDGER.exists() or LEDGER.stat().st_size == 0:
        return {"files": {}}

    try:
        data = json.loads(LEDGER.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"files": {}}
        data.setdefault("files", {})
        return data
    except json.JSONDecodeError:
        print(f"[WARNING] Invalid ledger: {LEDGER}")
        return {"files": {}}


def save_ledger(ledger: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    temp = LEDGER.with_suffix(".tmp")
    temp.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(LEDGER)


def append_jsonl(record: dict) -> None:
    RAW_RECORDS.parent.mkdir(parents=True, exist_ok=True)
    with RAW_RECORDS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def unique_destination(folder: Path, original_name: str) -> Path:
    destination = folder / original_name
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 2

    while True:
        candidate = folder / f"{stem}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def process_image(path: Path, ledger: dict) -> str:
    file_hash = sha256_file(path)

    existing = ledger["files"].get(file_hash)
    if existing:
        print(f"[SKIP] Already processed: {path.name}")
        return "skipped"

    print(f"[PROCESS] {path.name}")

    try:
        raw_analysis = analyze_chart(path)
        trusted_analysis = validate_analysis(raw_analysis)

        record_id = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        record = {
            "schema_version": "2.1",
            "record_id": record_id,
            "source": {
                "original_filename": path.name,
                "sha256": file_hash,
                "processed_at": now_iso(),
            },
            "teacher_analysis": raw_analysis,
            "trusted_analysis": trusted_analysis,
        }

        append_jsonl(record)

        quality = trusted_analysis.get("quality", {})
        confidence = float(quality.get("overall_confidence", 0.0))
        needs_review = bool(quality.get("needs_review", True))

        if needs_review or confidence < 0.85:
            destination_folder = REVIEW
            status = "review"
        else:
            destination_folder = PROCESSED
            status = "processed"

        destination = unique_destination(destination_folder, path.name)
        destination_folder.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))

        ledger["files"][file_hash] = {
            "record_id": record_id,
            "original_name": path.name,
            "processed_at": now_iso(),
            "status": status,
            "destination": str(destination),
        }
        save_ledger(ledger)

        print(f"[OK] {path.name} -> {status}")
        return status

    except Exception as exc:
        # Important: API/network/model errors do NOT move the image to failed.
        # The image stays in incoming so it can be retried.
        print(f"[ERROR] {path.name}: {exc}")
        return "error"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan charts/incoming for new chart images."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process the current incoming folder once.",
    )
    parser.parse_args()

    for folder in (INCOMING, PROCESSED, FAILED, REVIEW):
        folder.mkdir(parents=True, exist_ok=True)

    ledger = load_ledger()

    images = sorted(
        p for p in INCOMING.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print("[INFO] No new chart images found.")
        return

    print(f"[INFO] Found {len(images)} image(s).")

    counts = {
        "processed": 0,
        "review": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
    }

    for image in images:
        result = process_image(image, ledger)
        counts[result] = counts.get(result, 0) + 1

    print()
    print("========== SUMMARY ==========")

    if counts["processed"] > 0 or counts["review"] > 0:
        backup_records()

    for key, value in counts.items():
        print(f"{key:>10}: {value}")


def backup_records():
    source = Path("data/raw/chart_records.jsonl")
    backup_dir = Path("data/backup")

    if not source.exists() or source.stat().st_size == 0:
        return

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = backup_dir / f"chart_records_{timestamp}.jsonl"

    shutil.copy2(source, destination)

    print(f"[BACKUP] {destination}")


if __name__ == "__main__":
    main()
