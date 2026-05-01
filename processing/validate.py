"""Authored by: Rohan."""

import logging
import os
import sys

import pandas as pd
import great_expectations as gx
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import get_session
from config.settings import INTENT_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_validation_dataframe(session):
    """Read every transcript joined with its call's intent label into a DataFrame."""
    raw_query = text(
        "SELECT t.call_id, t.cleaned_transcript, c.intent_label "
        "FROM transcripts t "
        "JOIN calls c ON t.call_id = c.id"
    )
    rows = session.execute(raw_query).fetchall()
    return pd.DataFrame(rows, columns=["call_id", "cleaned_transcript", "intent_label"])


def run_validation():
    """Run the Great Expectations suite over the cleaned transcripts table."""
    with get_session() as session:
        logger.info("Fetching transcripts for Great Expectations validation")
        validation_dataframe = fetch_validation_dataframe(session)
        logger.info(f"Loaded {len(validation_dataframe)} rows for validation")

        validation_dataframe["intent_label"] = validation_dataframe["intent_label"].str.strip()

    context = gx.get_context(mode="file", project_root_dir="gx")

    for name in ("voiceintent_validation",):
        try:
            context.validation_definitions.delete(name)
        except (KeyError, Exception):
            pass
    for name in ("voiceintent_transcript_suite",):
        try:
            context.suites.delete(name)
        except (KeyError, Exception):
            pass
    try:
        context.data_sources.delete("voiceintent_pandas_source")
    except (KeyError, Exception):
        pass

    data_source = context.data_sources.add_pandas(name="voiceintent_pandas_source")
    data_asset = data_source.add_dataframe_asset(name="transcripts_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")

    suite = context.suites.add(gx.ExpectationSuite(name="voiceintent_transcript_suite"))

    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="cleaned_transcript"))
    suite.add_expectation(gx.expectations.ExpectColumnValueLengthsToBeBetween(
        column="cleaned_transcript", min_value=5, max_value=500
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="intent_label", value_set=INTENT_NAMES
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="call_id"))
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=1))

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(name="voiceintent_validation", data=batch_definition, suite=suite)
    )

    results = validation_definition.run(batch_parameters={"dataframe": validation_dataframe})

    if not results.success:
        failed = [r.expectation_config.type for r in results.results if not r.success]
        failure_summary = ", ".join(failed)
        logger.error(f"Validation FAILED: {failure_summary}")
        raise ValueError(f"Data quality validation failed. Failed expectations: {failure_summary}")

    logger.info("Great Expectations validation passed")
    return True


if __name__ == "__main__":
    run_validation()
