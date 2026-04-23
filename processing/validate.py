# AI620 VoiceIntent: Great Expectations Validation

import logging

import pandas as pd
import great_expectations as gx
from sqlalchemy import text

from storage.db import get_session

# Configuring the logger for this module

logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# All 77 intent labels from the BANKING77 dataset

BANKING77_INTENTS = ["activate_my_card",
                     "age_limit",
                     "apple_pay_or_google_pay",
                     "atm_support",
                     "automatic_top_up",
                     "balance_not_updated_after_bank_transfer",
                     "balance_not_updated_after_cheque_or_cash_deposit",
                     "beneficiary_not_allowed",
                     "cancel_transfer",
                     "card_about_to_expire",
                     "card_acceptance",
                     "card_arrival",
                     "card_delivery_estimate",
                     "card_linking",
                     "card_not_working",
                     "card_payment_fee_charged",
                     "card_payment_not_recognised",
                     "card_payment_wrong_exchange_rate",
                     "card_swallowed",
                     "cash_withdrawal_charge",
                     "cash_withdrawal_not_recognised",
                     "change_pin",
                     "compromised_card",
                     "contactless_not_working",
                     "country_support",
                     "declined_card_payment",
                     "declined_cash_withdrawal",
                     "declined_transfer",
                     "direct_debit_payment_not_recognised",
                     "disposable_card_limits",
                     "edit_personal_details",
                     "exchange_charge",
                     "exchange_rate",
                     "exchange_via_app",
                     "extra_charge_on_statement",
                     "failed_transfer",
                     "fiat_currency_support",
                     "get_disposable_virtual_card",
                     "get_physical_card",
                     "getting_spare_card",
                     "getting_virtual_card",
                     "lost_or_stolen_card",
                     "lost_or_stolen_phone",
                     "order_physical_card",
                     "passcode_forgotten",
                     "pending_card_payment",
                     "pending_cash_withdrawal",
                     "pending_top_up",
                     "pending_transfer",
                     "pin_blocked",
                     "receiving_money",
                     "refund_not_showing_up",
                     "request_refund",
                     "reverted_card_payment?",
                     "romania_transfer",
                     "scam_customer",
                     "supported_cards_and_currencies",
                     "terminate_account",
                     "top_up_by_bank_transfer_charge",
                     "top_up_by_card_charge",
                     "top_up_by_cash_or_cheque",
                     "top_up_failed",
                     "top_up_limits",
                     "top_up_reverted",
                     "topping_up_by_card",
                     "transaction_charged_twice",
                     "transfer_fee_charged",
                     "transfer_into_account",
                     "transfer_not_received_by_recipient",
                     "transfer_timing",
                     "unable_to_verify_identity",
                     "verify_my_identity",
                     "verify_source_of_funds",
                     "verify_top_up",
                     "virtual_card_not_working",
                     "visa_or_mastercard",
                     "why_verify_identity",
                     "wrong_amount_of_cash_received",
                     "wrong_exchange_rate_for_cash_withdrawal",]

# Querying the transcripts joined with calls to build the validation DataFrame

def fetch_validation_dataframe(session):

    raw_query = text("SELECT t.call_id, t.cleaned_transcript, c.intent_label "
                     "FROM transcripts t "
                     "JOIN calls c ON t.call_id = c.id")

    rows = session.execute(raw_query).fetchall()
    validation_dataframe = pd.DataFrame(rows, columns = ["call_id", "cleaned_transcript", "intent_label"])
    return validation_dataframe

# Running the Great Expectations suite and raising on any failure so Prefect marks the task as FAILED

def run_validation():

    with get_session() as session:

        logger.info("Fetching transcripts for Great Expectations validation")
        validation_dataframe = fetch_validation_dataframe(session)
        logger.info(f"Loaded {len(validation_dataframe)} rows for validation")

    data_context = gx.get_context() # Building a GX context from the gx/ folder at project root

    # Registering a pandas datasource pointing at the validation DataFrame

    datasource = data_context.sources.add_or_update_pandas(name = "voiceintent_pandas_source")
    data_asset = datasource.add_dataframe_asset(name = "transcripts_asset")
    batch_request = data_asset.build_batch_request(dataframe = validation_dataframe)

    # Getting or creating the expectation suite

    suite_name = "voiceintent_transcript_suite"

    try:
        expectation_suite = data_context.get_expectation_suite(suite_name)
        logger.info(f"Loaded existing suite '{suite_name}'")
    except Exception:
        expectation_suite = data_context.add_expectation_suite(suite_name)
        logger.info(f"Created new suite '{suite_name}'")

    validator = data_context.get_validator(batch_request = batch_request,
                                           expectation_suite_name = suite_name)

    # Adding the 5 required expectations

    validator.expect_column_values_to_not_be_null("cleaned_transcript") # 1: no nulls in cleaned_transcript

    validator.expect_column_value_lengths_to_be_between("cleaned_transcript",
                                                         min_value = 5,
                                                         max_value = 500) # 2: transcript length between 5 and 500 characters

    validator.expect_column_values_to_be_in_set("intent_label", BANKING77_INTENTS) # 3: only known BANKING77 labels allowed

    validator.expect_column_values_to_not_be_null("call_id") # 4: every transcript must link back to a call

    validator.expect_table_row_count_to_be_between(min_value = 1) # 5: catches silent empty result bugs

    validator.save_expectation_suite(discard_failed_expectations = False) # Saving the suite to gx/expectations/

    # Running the checkpoint and saving data docs to gx/uncommitted/data_docs/

    checkpoint = data_context.add_or_update_checkpoint(name = "voiceintent_checkpoint",
                                                       validator = validator)

    checkpoint_result = checkpoint.run()

    if not checkpoint_result.success: # Collecting which expectations failed for a clear error message

        failed_expectation_list = []

        for validation_result in checkpoint_result.run_results.values():
            for expectation_result in validation_result["validation_result"]["results"]:
                if not expectation_result["success"]:
                    failed_expectation_list.append(expectation_result["expectation_config"]["expectation_type"])

        failure_summary = ", ".join(failed_expectation_list)
        logger.error(f"Validation FAILED: failed expectations: {failure_summary}")

        raise ValueError(f"Data quality validation failed. " # Raising so Prefect marks validate_task as FAILED and blocks downstream tasks
                         f"Failed expectations: {failure_summary}. "
                         f"Check gx/uncommitted/data_docs/ for the full report.")

    logger.info("Great Expectations validation passed: all expectations met")
    return True


if __name__ == "__main__":
    run_validation()
