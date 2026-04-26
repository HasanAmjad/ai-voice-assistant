<<<<<<< Updated upstream
import logging
import pandas as pd
import great_expectations as gx
from sqlalchemy import text
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BANKING77_INTENTS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_blocked", "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "dispute_purchase",
    "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "freeze_account", "get_disposable_virtual_card", "get_physical_card",
    "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
    "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money", "refund_not_showing_up",
    "request_refund", "reverted_card_payment", "savings_account",
    "seen_a_sign_recently", "send_invoice", "share_balance",
    "sharing_personal_information", "show_purchases", "supported_cards_and_currencies",
    "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity",
]


def fetch_validation_dataframe(session):
    raw_query = text("SELECT t.call_id, t.cleaned_transcript, c.intent_label "
                     "FROM transcripts t "
                     "JOIN calls c ON t.call_id = c.id")
    rows = session.execute(raw_query).fetchall()
    return pd.DataFrame(rows, columns=["call_id", "cleaned_transcript", "intent_label"])

def run_validation():
    with get_session() as session:
        logger.info("Fetching transcripts for Great Expectations validation")
        validation_dataframe = fetch_validation_dataframe(session)
        logger.info(f"Loaded {len(validation_dataframe)} rows for validation")

        # debug lines
        validation_dataframe['intent_label'] = validation_dataframe['intent_label'].str.strip()
        print("UNIQUE INTENTS:", sorted(validation_dataframe['intent_label'].unique()))
        print("SAMPLE BYTES:", [validation_dataframe['intent_label'].iloc[0].encode()])

    context = gx.get_context(mode="file", project_root_dir="gx")
    data_source = context.data_sources.add_pandas(name="voiceintent_pandas_source")
    data_asset = data_source.add_dataframe_asset(name="transcripts_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")
    
    # Ensure dataframe is passed here
    batch = batch_definition.get_batch(batch_parameters={"dataframe": validation_dataframe})

    suite = context.suites.add(gx.ExpectationSuite(name="voiceintent_transcript_suite"))

    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="cleaned_transcript"))
    suite.add_expectation(gx.expectations.ExpectColumnValueLengthsToBeBetween(column="cleaned_transcript", min_value=5, max_value=500))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="intent_label", value_set=BANKING77_INTENTS))
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
=======
def run_validation():
    print("⚠️  STUB: run_validation() not implemented yet")
    print("    Expected: Run Great Expectations validation suite")
    print("")
>>>>>>> Stashed changes
