from sqlalchemy import (
    Column, Integer, Text, Float, VARCHAR,
    TIMESTAMP, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Call(Base):
    __tablename__ = "calls"
    __table_args__ = (UniqueConstraint("audio_file_path"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    audio_file_path = Column(Text, unique=True, nullable=False)
    intent_label = Column(VARCHAR(80), nullable=False)
    split = Column(VARCHAR(10), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    transcripts = relationship("Transcript", back_populates="call")
    predictions = relationship("Prediction", back_populates="call")


class Transcript(Base):
    __tablename__ = "transcripts"
    __table_args__ = (UniqueConstraint("call_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=False)
    raw_transcript = Column(Text)
    cleaned_transcript = Column(Text)
    transcribed_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    call = relationship("Call", back_populates="transcripts")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=True)
    predicted_intent = Column(VARCHAR(80))
    confidence_score = Column(Float)
    cleaned_transcript = Column(Text)
    model_version = Column(VARCHAR(40))
    predicted_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    call = relationship("Call", back_populates="predictions")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(VARCHAR(40), nullable=False)
    accuracy = Column(Float)
    macro_f1 = Column(Float)
    training_samples = Column(Integer)
    data_hash = Column(Text)
    drift_score = Column(Float)
    trained_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
