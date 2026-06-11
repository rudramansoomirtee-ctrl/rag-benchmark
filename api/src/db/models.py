"""SQLAlchemy models — matches the five-table schema in the design doc."""
from datetime import datetime
from sqlalchemy import (
    Integer, String, Text, JSON, DateTime, Boolean, Float, Numeric,
    ForeignKey, UniqueConstraint, Index, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)

    runs: Mapped[list["Run"]] = relationship(back_populates="experiment", cascade="all, delete")


class Query(Base):
    __tablename__ = "queries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(Text, nullable=False)         # 'multihop' | 'ragtruth'
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    split: Mapped[str] = mapped_column(Text, nullable=False)           # 'calibration' | 'eval'
    task_type: Mapped[str | None] = mapped_column(Text)                # 'qa' | 'summary' | 'data2txt'
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str | None] = mapped_column(Text)
    relevant_chunk_ids: Mapped[list] = mapped_column(JSONB, nullable=False)
    query_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB)

    __table_args__ = (
        UniqueConstraint("dataset", "external_id"),
        Index("ix_queries_dataset_split", "dataset", "split"),
    )


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB)

    __table_args__ = (UniqueConstraint("dataset", "external_id"),)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    system: Mapped[str] = mapped_column(Text, nullable=False)
    query_id: Mapped[int] = mapped_column(ForeignKey("queries.id"))
    retrieved_chunk_ids: Mapped[list] = mapped_column(JSONB, nullable=False)  # final answering context
    all_retrieved_chunk_ids: Mapped[list | None] = mapped_column(JSONB)       # evidence ever seen (retrieval-ceiling)
    answer: Mapped[str | None] = mapped_column(Text)
    hhem_score: Mapped[float | None] = mapped_column(Float)
    flagged: Mapped[bool | None] = mapped_column(Boolean)
    n_steps: Mapped[int | None] = mapped_column(Integer)
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 6))
    is_correct: Mapped[bool | None] = mapped_column(Boolean)        # primary: contains_match
    llm_judge_label: Mapped[str | None] = mapped_column(Text)       # secondary: CRAG rubric
    phoenix_trace_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    experiment: Mapped["Experiment"] = relationship(back_populates="runs")

    __table_args__ = (
        UniqueConstraint("experiment_id", "system", "query_id"),
        Index("ix_runs_experiment_system", "experiment_id", "system"),
    )


class Metric(Base):
    __tablename__ = "metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    system: Mapped[str] = mapped_column(Text, nullable=False)
    dataset: Mapped[str] = mapped_column(Text, nullable=False)
    n_queries: Mapped[int] = mapped_column(Integer, nullable=False)
    precision_at_5: Mapped[float | None] = mapped_column(Float)
    recall_at_5: Mapped[float | None] = mapped_column(Float)
    avg_faithfulness: Mapped[float | None] = mapped_column(Float)
    pct_flagged: Mapped[float | None] = mapped_column(Float)
    avg_trajectory_length: Mapped[float | None] = mapped_column(Float)
    pct_failed: Mapped[float | None] = mapped_column(Float)         # fraction of runs that errored (answer IS NULL); these count as wrong in `accuracy`
    accuracy: Mapped[float | None] = mapped_column(Float)           # primary: contains_match
    accuracy_exact: Mapped[float | None] = mapped_column(Float)     # secondary: normalized exact-match
    avg_token_f1: Mapped[float | None] = mapped_column(Float)       # secondary: mean SQuAD-style token F1
    crag_score: Mapped[float | None] = mapped_column(Float)         # secondary: mean CRAG truthfulness
    total_cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 4))
    cost_per_correct: Mapped[float | None] = mapped_column(Numeric(10, 6))
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("experiment_id", "system", "dataset"),)
