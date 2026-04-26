import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from storage.db import get_session

BANKING77_INTENTS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_blocked", "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
    "card_swallowed", "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
    "change_pin", "compromised_card", "contactless_not_working", "country_support",
    "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "dispute_purchase", "edit_personal_details",
    "exchange_charge", "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
    "failed_transfer", "fiat_currency_support", "freeze_account", "get_disposable_virtual_card",
    "get_physical_card", "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
    "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
    "receiving_money", "refund_not_showing_up", "request_refund", "reverted_card_payment",
    "savings_account", "seen_a_sign_recently", "send_invoice", "share_balance",
    "sharing_personal_information", "show_purchases", "supported_cards_and_currencies",
    "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity"
]

with get_session() as session:
    rows = session.execute(text(
        "SELECT DISTINCT c.intent_label FROM calls c JOIN transcripts t ON c.id = t.call_id ORDER BY c.intent_label"
    )).fetchall()

db_labels = set(r[0] for r in rows)
known_labels = set(BANKING77_INTENTS)

print("=== In DB but NOT in BANKING77_INTENTS ===")
for label in sorted(db_labels - known_labels):
    print(f"  MISSING: '{label}'")

print("\n=== In BANKING77_INTENTS but NOT in DB ===")
for label in sorted(known_labels - db_labels):
    print(f"  EXTRA: '{label}'")

print(f"\nDB total: {len(db_labels)} | List total: {len(known_labels)}")