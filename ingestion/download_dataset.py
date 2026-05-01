"""Authored by: Hasan."""

import os
import pandas as pd
import huggingface_hub
from datasets import load_dataset
from dotenv import load_dotenv
from config.settings import RAW_DIR, HF_TOKEN

load_dotenv()


def download():
    """Download the BANKING77 train and test splits from HuggingFace into RAW_DIR."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    train_path = RAW_DIR / "banking77_train.csv"
    test_path = RAW_DIR / "banking77_test.csv"

    if train_path.exists() and test_path.exists():
        print("Dataset already downloaded, skipping.")
        return

    huggingface_hub.login(token=HF_TOKEN)
    print("Downloading BANKING77 from HuggingFace...")

    ds = load_dataset("mteb/banking77")

    for split in ["train", "test"]:
        df = ds[split].to_pandas()
        path = RAW_DIR / f"banking77_{split}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} rows -> {path}")


if __name__ == "__main__":
    download()
