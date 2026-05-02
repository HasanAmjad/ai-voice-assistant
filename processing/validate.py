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


DS_NAME = "voiceintent_pandas_source"
ASSET_NAME = "transcripts_asset"
BATCH_NAME = "batch"
SUITE_NAME = "voiceintent_transcript_suite"
VD_NAME = "voiceintent_validation"


def _get_or_add_datasource(context):
    """Return the named pandas data source, creating it only if missing."""
    try:
        return context.data_sources.get(DS_NAME)
    except Exception:
        return context.data_sources.add_pandas(name=DS_NAME)


def _get_or_add_asset(data_source):
    """Return the named dataframe asset on `data_source`, creating it only if missing."""
    try:
        return data_source.get_asset(ASSET_NAME)
    except Exception:
        return data_source.add_dataframe_asset(name=ASSET_NAME)


def _get_or_add_batch_definition(data_asset):
    """Return the named whole-dataframe batch definition, creating it only if missing."""
    try:
        existing = [bd for bd in (getattr(data_asset, "batch_definitions", []) or []) if bd.name == BATCH_NAME]
        if existing:
            return existing[0]
    except Exception:
        pass
    return data_asset.add_batch_definition_whole_dataframe(BATCH_NAME)


def _get_or_add_suite(context):
    """Return the named expectation suite (with all 5 expectations), creating it only if missing."""
    try:
        return context.suites.get(SUITE_NAME)
    except Exception:
        suite = context.suites.add(gx.ExpectationSuite(name=SUITE_NAME))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="cleaned_transcript"))
        suite.add_expectation(gx.expectations.ExpectColumnValueLengthsToBeBetween(
            column="cleaned_transcript", min_value=5, max_value=500
        ))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
            column="intent_label", value_set=INTENT_NAMES
        ))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="call_id"))
        suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=1))
        return suite


def _get_or_add_validation_definition(context, batch_definition, suite):
    """Return the validation definition tying the batch to the suite, creating it only if missing."""
    try:
        return context.validation_definitions.get(VD_NAME)
    except Exception:
        return context.validation_definitions.add(
            gx.ValidationDefinition(name=VD_NAME, data=batch_definition, suite=suite)
        )


def run_validation():
    """Run the Great Expectations suite over the cleaned transcripts table.

    The GX entities (data source, asset, batch definition, suite, validation definition)
    are created on the first run and reused on every subsequent run, so the persisted
    `gx/great_expectations.yml`, suite JSON, and validation-definition JSON stay stable.
    The validation itself is executed every call against the current dataframe.
    """
    with get_session() as session:
        logger.info("Fetching transcripts for Great Expectations validation")
        validation_dataframe = fetch_validation_dataframe(session)
        logger.info(f"Loaded {len(validation_dataframe)} rows for validation")

        validation_dataframe["intent_label"] = validation_dataframe["intent_label"].str.strip()

    context = gx.get_context(mode="file", project_root_dir="gx")

    data_source = _get_or_add_datasource(context)
    data_asset = _get_or_add_asset(data_source)
    batch_definition = _get_or_add_batch_definition(data_asset)
    suite = _get_or_add_suite(context)
    validation_definition = _get_or_add_validation_definition(context, batch_definition, suite)

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
