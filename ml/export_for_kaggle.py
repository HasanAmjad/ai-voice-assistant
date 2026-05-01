import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
from storage.db import get_session
from sqlalchemy import text

with get_session() as session:
    result = session.execute(text("""
        SELECT t.cleaned_transcript, c.intent_label, c.split
        FROM transcripts t
        JOIN calls c ON t.call_id = c.id
        WHERE t.cleaned_transcript IS NOT NULL
          AND t.cleaned_transcript != '[INAUDIBLE]'
    """))
    df = pd.DataFrame(result.fetchall(), columns=['cleaned_transcript', 'intent_label', 'split'])

print(f"Total: {len(df):,} | Train: {len(df[df.split=='train']):,} | Test: {len(df[df.split=='test']):,}")
print(f"Intents: {df['intent_label'].nunique()}")
df.to_csv('ml/whisper_transcripts_export.csv', index=False)
print("Saved ml/whisper_transcripts_export.csv")
