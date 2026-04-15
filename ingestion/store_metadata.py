from storage.db import get_session, init_db
from storage.models import Call
from config.settings import AUDIO_DIR, INTENT_NAMES

def store():
    init_db()
    inserted, skipped = 0, 0

    with get_session() as session:
        for mp3 in AUDIO_DIR.rglob("*.mp3"):
            # skip zero-byte sentinel files
            if mp3.stat().st_size == 0:
                continue

            parts = mp3.parts
            split       = parts[-3]          # "train" or "test"
            label_num   = int(parts[-2])     # folder number e.g. 4
            intent_name = INTENT_NAMES[label_num]  # maps to proper name
            path        = str(mp3.resolve())

            exists = session.query(Call).filter_by(audio_file_path=path).first()
            if exists:
                skipped += 1
                continue

            session.add(Call(
                audio_file_path=path,
                intent_label=intent_name,   # stores "card_arrival" not "4"
                split=split
            ))
            inserted += 1

    print(f"Done. Inserted: {inserted} | Skipped (already existed): {skipped}")

if __name__ == "__main__":
    store()