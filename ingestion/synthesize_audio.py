"""Authored by: Hasan."""

import time
import pandas as pd
from gtts import gTTS
from config.settings import RAW_DIR, AUDIO_DIR


def synthesize():
    """Convert every BANKING77 text row into an .mp3 stored under AUDIO_DIR/<split>/<intent>/."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    total, skipped, failed = 0, 0, 0

    for split in ["train", "test"]:
        df = pd.read_csv(RAW_DIR / f"banking77_{split}.csv")
        total_rows = len(df)
        print(f"\nProcessing '{split}' split - {total_rows} rows")

        for idx, row in df.iterrows():
            intent = row["label"] if "label" in df.columns else row["category"]
            text = row["text"]

            folder = AUDIO_DIR / split / str(intent)
            folder.mkdir(parents=True, exist_ok=True)
            filepath = folder / f"{idx}.mp3"

            if filepath.exists():
                skipped += 1
                continue

            try:
                tts = gTTS(text=text, lang="en")
                tts.save(str(filepath))
                total += 1
                time.sleep(0.5)

                if (total + skipped) % 100 == 0:
                    print(f"  [{split}] Row {idx}/{total_rows} | Synthesised: {total} | Skipped: {skipped} | Failed: {failed}")

            except Exception as e:
                if "429" in str(e):
                    print(f"  Rate limited at row {idx}. Waiting 60 seconds...")
                    time.sleep(60)
                    try:
                        tts = gTTS(text=text, lang="en")
                        tts.save(str(filepath))
                        total += 1
                    except Exception as e2:
                        print(f"  ERROR row {idx}: {e2}")
                        filepath.touch()
                        failed += 1
                else:
                    print(f"  ERROR row {idx}: {e}")
                    filepath.touch()
                    failed += 1

    print(f"\nDone. Synthesised: {total} | Skipped: {skipped} | Failed: {failed}")


if __name__ == "__main__":
    synthesize()
