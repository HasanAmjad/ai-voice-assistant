from storage.db import get_session, init_db
from storage.models import Call
from config.settings import AUDIO_DIR, INTENT_NAMES

def store():
    init_db()
    inserted, skipped, zero_byte = 0, 0, 0

    with get_session() as session:
        for mp3 in AUDIO_DIR.rglob("*.mp3"):

            # skip zero-byte sentinel files
            if mp3.stat().st_size == 0:
                zero_byte += 1
                continue

            parts = mp3.parts
            split      = parts[-3]           # "train" or "test"
            label_num  = int(parts[-2])      # folder number e.g. 4
            intent     = INTENT_NAMES[label_num]  # maps to "card_arrival" etc.
            path       = str(mp3.resolve())

            exists = session.query(Call).filter_by(audio_file_path=path).first()
            if exists:
                skipped += 1
                continue

            session.add(Call(
                audio_file_path=path,
                intent_label=intent,
                split=split
            ))
            inserted += 1

        session.commit()

    print(f"\nDone.")
    print(f"  Inserted : {inserted}")
    print(f"  Skipped  : {skipped} (already in DB)")
    print(f"  Zero-byte: {zero_byte} (failed synthesis, skipped)")

if __name__ == "__main__":
    store()