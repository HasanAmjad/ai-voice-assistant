"""Authored by: Hasan."""

from storage.db import get_session, init_db
from storage.models import Call
from config.settings import AUDIO_DIR, INTENT_NAMES


def store():
    """Walk AUDIO_DIR and insert one row per non-empty .mp3 into the calls table."""
    init_db()
    inserted, skipped, zero_byte, ignored = 0, 0, 0, 0

    with get_session() as session:
        for mp3 in AUDIO_DIR.rglob("*.mp3"):

            if mp3.stat().st_size == 0:
                zero_byte += 1
                continue

            parts = mp3.parts
            split = parts[-3]
            if split not in ("train", "test"):
                ignored += 1
                continue
            try:
                label_num = int(parts[-2])
            except ValueError:
                ignored += 1
                continue
            intent = INTENT_NAMES[label_num]
            path = str(mp3.resolve())

            exists = session.query(Call).filter_by(audio_file_path=path).first()
            if exists:
                skipped += 1
                continue

            session.add(Call(
                audio_file_path=path,
                intent_label=intent,
                split=split,
            ))
            inserted += 1

        session.commit()

    print(f"\nDone.")
    print(f"  Inserted : {inserted}")
    print(f"  Skipped  : {skipped} (already in DB)")
    print(f"  Zero-byte: {zero_byte} (failed synthesis, skipped)")
    print(f"  Ignored  : {ignored} (outside data/audio/train|test/<intent_idx>/ layout, e.g. samples/)")


if __name__ == "__main__":
    store()
