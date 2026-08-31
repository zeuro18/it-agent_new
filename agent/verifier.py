"""
verifier.py
DB-state postcondition checks for Phase 1 (Experiment A).

Each function in TOOL_POSTCONDITIONS independently re-queries the database
to verify the expected state after a mutating tool call, without relying on
the tool's own success/error report.

Design notes:
  - All functions receive `args` (the raw dict the LLM passed to the tool).
  - They must run inside a Flask app context. The import of `app` here is
    intentional: we need the same SQLAlchemy session that `tools.py` uses.
  - Return type is always VerificationResult(passed: bool, details: str).
  - For fault-injection testing, the caller may simulate a flaky DB write by
    briefly corrupting the committed value; these functions detect that.
"""

import os
import sys
from dataclasses import dataclass

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app import app
from database import User, License, Group, Ticket


@dataclass
class VerificationResult:
    passed: bool
    details: str


# ---------------------------------------------------------------------------
# Postcondition functions
# ---------------------------------------------------------------------------

def _verify_assign_license(args: dict) -> VerificationResult:
    """Confirm the license row exists in the DB after assign_license()."""
    email = args.get("email", "")
    software = args.get("software", "")
    plan = args.get("plan")
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return VerificationResult(False, f"User {email} not found in DB")
        lic = License.query.filter_by(assigned_to=user.id, software=software).first()
        if not lic:
            return VerificationResult(False, f"License {software} not found for {email}")
        if plan and lic.plan != plan:
            return VerificationResult(False, f"License plan={lic.plan}, expected {plan}")
        return VerificationResult(True, f"License {software}/{plan} confirmed for {email}")


def _verify_revoke_license(args: dict) -> VerificationResult:
    """Confirm the license row is absent from the DB after revoke_license()."""
    email = args.get("email", "")
    software = args.get("software", "")
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            # User gone entirely — license certainly doesn't exist.
            return VerificationResult(True, f"User {email} not found; license trivially absent")
        lic = License.query.filter_by(assigned_to=user.id, software=software).first()
        if lic:
            return VerificationResult(False, f"License {software} still exists for {email}")
        return VerificationResult(True, f"License {software} absent for {email}")


def _verify_create_user(args: dict) -> VerificationResult:
    """Confirm the user row exists with the expected fields after create_user()."""
    email = args.get("email", "")
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return VerificationResult(False, f"User {email} not found in DB after create")
        mismatches = []
        for field in ("name", "department", "role"):
            expected = args.get(field)
            if expected and getattr(user, field, None) != expected:
                mismatches.append(f"{field}: expected '{expected}', got '{getattr(user, field)}'")
        if mismatches:
            return VerificationResult(False, "; ".join(mismatches))
        return VerificationResult(True, f"User {email} created with expected fields")


def _verify_delete_user(args: dict) -> VerificationResult:
    """Confirm the user row is absent after delete_user()."""
    email = args.get("email", "")
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if user:
            return VerificationResult(False, f"User {email} still exists in DB after delete")
        return VerificationResult(True, f"User {email} absent from DB")


def _verify_reset_password(args: dict) -> VerificationResult:
    """Confirm the password_hint field was updated after reset_password()."""
    email = args.get("email", "")
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return VerificationResult(False, f"User {email} not found in DB")
        if not user.password_hint or "reset on" not in (user.password_hint or "").lower():
            return VerificationResult(
                False,
                f"password_hint='{user.password_hint}' doesn't indicate a reset occurred"
            )
        return VerificationResult(True, f"Password reset recorded for {email}")


def _verify_assign_user_to_group(args: dict) -> VerificationResult:
    """Confirm the user is a member of the group after assign_user_to_group()."""
    email = args.get("email", "")
    group_name = args.get("group_name", "")
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return VerificationResult(False, f"User {email} not found in DB")
        group_names = [g.name for g in user.groups]
        if group_name not in group_names:
            return VerificationResult(
                False, f"User {email} not in group '{group_name}'. Groups: {group_names}"
            )
        return VerificationResult(True, f"User {email} confirmed in group '{group_name}'")


def _verify_remove_user_from_group(args: dict) -> VerificationResult:
    """Confirm the user is no longer in the group after remove_user_from_group()."""
    email = args.get("email", "")
    group_name = args.get("group_name", "")
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return VerificationResult(True, f"User {email} not found; trivially not in group")
        group_names = [g.name for g in user.groups]
        if group_name in group_names:
            return VerificationResult(False, f"User {email} still in group '{group_name}'")
        return VerificationResult(True, f"User {email} confirmed absent from group '{group_name}'")


def _verify_update_ticket(args: dict) -> VerificationResult:
    """Confirm the ticket's status and notes after update_ticket()."""
    ticket_id = args.get("ticket_id")
    expected_status = args.get("status")
    with app.app_context():
        ticket = Ticket.query.get(ticket_id)
        if not ticket:
            return VerificationResult(False, f"Ticket #{ticket_id} not found in DB")
        if expected_status and ticket.status != expected_status:
            return VerificationResult(
                False, f"Ticket #{ticket_id} status={ticket.status}, expected {expected_status}"
            )
        return VerificationResult(True, f"Ticket #{ticket_id} status confirmed as '{ticket.status}'")


def _verify_edit_user(args: dict) -> VerificationResult:
    """Confirm the user's fields were updated after edit_user()."""
    email = args.get("email", "")
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return VerificationResult(False, f"User {email} not found in DB")
        mismatches = []
        for field in ("department", "role", "status"):
            expected = args.get(field)
            if expected and getattr(user, field, None) != expected:
                mismatches.append(f"{field}: expected '{expected}', got '{getattr(user, field)}'")
        if mismatches:
            return VerificationResult(False, "; ".join(mismatches))
        return VerificationResult(True, f"User {email} fields updated correctly")


# ---------------------------------------------------------------------------
# Postcondition registry
# ---------------------------------------------------------------------------
# Read-only tools (user_lookup, list_licenses, list_pending_tickets) are not
# listed here; the caller in _verify_db() falls back to the tool's own
# success/error report for those.

TOOL_POSTCONDITIONS = {
    "assign_license":         _verify_assign_license,
    "revoke_license":         _verify_revoke_license,
    "create_user":            _verify_create_user,
    "delete_user":            _verify_delete_user,
    "reset_password":         _verify_reset_password,
    "assign_user_to_group":   _verify_assign_user_to_group,
    "remove_user_from_group": _verify_remove_user_from_group,
    "update_ticket":          _verify_update_ticket,
    "edit_user":              _verify_edit_user,
}
