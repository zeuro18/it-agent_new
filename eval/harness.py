"""
harness.py
──────────
Evaluation harness for the IT Agent.

Runs tasks from tasks_bank.json against the agent, validates DB state
against ground truth, detects side effects, and records metrics.

Usage:
    python eval/harness.py                          # Run all tasks with default config
    python eval/harness.py --config baseline        # No tools, no RAG (browser only concept)
    python eval/harness.py --config tools_only      # Tools, no RAG
    python eval/harness.py --config dense            # Tools + Dense RAG
    python eval/harness.py --config bm25             # Tools + BM25 RAG
    python eval/harness.py --config hybrid           # Tools + Hybrid RAG
    python eval/harness.py --fast                   # Run smoke-test subset only
"""

import os
import sys
import json
import time
import shutil
import copy
from datetime import datetime
from pathlib import Path

# Ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'agent'))

from app import app
from database import db, User, License, AuditLog, Group, Ticket

TASKS_PATH = os.path.join(os.path.dirname(__file__), "tasks_bank.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── DB Snapshot & Reset ──────────────────────────────────────────────

def snapshot_db() -> dict:
    """Take a full snapshot of the DB state for side-effect detection."""
    with app.app_context():
        users = [{
            "id": u.id, "name": u.name, "email": u.email,
            "role": u.role, "department": u.department,
            "status": u.status, "password_hint": u.password_hint,
            "groups": sorted([g.name for g in u.groups])
        } for u in User.query.all()]

        licenses = [{
            "id": l.id, "software": l.software,
            "assigned_to": l.assigned_to, "plan": l.plan
        } for l in License.query.all()]

        tickets = [{
            "id": t.id, "created_for": t.created_for,
            "issue": t.issue, "priority": t.priority,
            "status": t.status, "notes": t.notes
        } for t in Ticket.query.all()]

        groups = [{
            "id": g.id, "name": g.name,
            "members": sorted([u.email for u in g.users])
        } for g in Group.query.all()]

        return {
            "users": users,
            "licenses": licenses,
            "tickets": tickets,
            "groups": groups,
            "audit_count": AuditLog.query.count()
        }


def reset_db():
    """Reset the database to the seed state by dropping and re-creating tables."""
    with app.app_context():
        db.drop_all()
        db.create_all()

        users = [
           User(name="Manas Mehta", email="manas@company.com", role="employee", status="active", department="Engineering"),
            User(name="ABCD", email="abcd@company.com", role="manager", status="active", department="HR"),
            User(name="EFGH", email="efgh@company.com", role="employee", status="inactive", department="Sales"),
            User(name="HIJK", email="hijk@company.com", role="employee", status="inactive", department="IT"),
            User(name="Nandini Menon", email="Nandini@company.com", role="employee", status="active", department="Legal")
        ]
        db.session.add_all(users)
        db.session.commit()

        licenses = [
            License(software="Microsoft 365", assigned_to=1, plan="Pro", assigned_date=datetime.utcnow()),
            License(software="Slack", assigned_to=1, plan="Business", assigned_date=datetime.utcnow()),
            License(software="Microsoft 365", assigned_to=2, plan="Pro", assigned_date=datetime.utcnow()),
        ]
        db.session.add_all(licenses)
        db.session.commit()

        groups = [
            Group(name="Developers", description="Engineering team members"),
            Group(name="Marketing", description="Marketing and Sales team members"),
            Group(name="HR", description="Human Resources"),
            Group(name="IT Admin", description="IT Administrators")
        ]
        db.session.add_all(groups)
        db.session.commit()

        users[0].groups.append(groups[0])  # Manas → Developers
        users[1].groups.append(groups[2])  # Pranav → HR
        db.session.commit()

        tickets = [
            Ticket(created_for=1, issue="Requesting access to GitHub Copilot", priority="Medium", status="Pending"),
            Ticket(created_for=2, issue="Need a new laptop", priority="High", status="Approved", notes="Processing order"),
            Ticket(created_for=3, issue="Cannot access Jira", priority="High", status="Resolved", notes="Reset password")
        ]
        db.session.add_all(tickets)
        db.session.commit()


# ── Validation Checks ────────────────────────────────────────────────

def validate_expected(expected: dict) -> dict:
    """
    Validate the expected DB state after a task.
    Returns {"passed": bool, "details": str}
    """
    check_type = expected.get("check")

    with app.app_context():
        if check_type == "user_exists":
            user = User.query.filter_by(email=expected["email"]).first()
            if not user:
                return {"passed": False, "details": f"User {expected['email']} not found"}
            for field in ["name", "department", "role", "status"]:
                if field in expected and getattr(user, field) != expected[field]:
                    return {"passed": False, "details": f"Expected {field}={expected[field]}, got {getattr(user, field)}"}
            return {"passed": True, "details": "User exists with expected fields"}

        elif check_type == "user_not_exists":
            user = User.query.filter_by(email=expected["email"]).first()
            if user:
                return {"passed": False, "details": f"User {expected['email']} still exists"}
            return {"passed": True, "details": "User correctly absent"}

        elif check_type == "user_field":
            user = User.query.filter_by(email=expected["email"]).first()
            if not user:
                return {"passed": False, "details": f"User {expected['email']} not found"}
            val = getattr(user, expected["field"], "")
            if expected.get("contains") and expected["contains"] not in (val or ""):
                return {"passed": False, "details": f"Field {expected['field']} = '{val}', expected to contain '{expected['contains']}'"}
            return {"passed": True, "details": f"Field check passed: {expected['field']}"}

        elif check_type == "license_exists":
            user = User.query.filter_by(email=expected["email"]).first()
            if not user:
                return {"passed": False, "details": f"User {expected['email']} not found"}
            lic = License.query.filter_by(assigned_to=user.id, software=expected["software"]).first()
            if not lic:
                return {"passed": False, "details": f"License {expected['software']} not found for {expected['email']}"}
            if "plan" in expected and lic.plan != expected["plan"]:
                return {"passed": False, "details": f"License plan={lic.plan}, expected {expected['plan']}"}
            return {"passed": True, "details": "License exists"}

        elif check_type == "license_not_exists":
            user = User.query.filter_by(email=expected["email"]).first()
            if not user:
                return {"passed": True, "details": "User not found, so license can't exist"}
            lic = License.query.filter_by(assigned_to=user.id, software=expected["software"]).first()
            if lic:
                return {"passed": False, "details": f"License {expected['software']} still exists"}
            return {"passed": True, "details": "License correctly absent"}

        elif check_type == "license_count":
            user = User.query.filter_by(email=expected["email"]).first()
            if not user:
                return {"passed": False, "details": f"User not found"}
            count = License.query.filter_by(assigned_to=user.id).count()
            if "min_count" in expected and count < expected["min_count"]:
                return {"passed": False, "details": f"License count={count}, expected >= {expected['min_count']}"}
            if "exact_count" in expected and count != expected["exact_count"]:
                return {"passed": False, "details": f"License count={count}, expected {expected['exact_count']}"}
            return {"passed": True, "details": f"License count check passed ({count})"}

        elif check_type == "user_in_group":
            user = User.query.filter_by(email=expected["email"]).first()
            if not user:
                return {"passed": False, "details": f"User not found"}
            group_names = [g.name for g in user.groups]
            if expected["group"] not in group_names:
                return {"passed": False, "details": f"User not in {expected['group']}. In: {group_names}"}
            return {"passed": True, "details": f"User in {expected['group']}"}

        elif check_type == "user_not_in_group":
            user = User.query.filter_by(email=expected["email"]).first()
            if not user:
                return {"passed": True, "details": "User not found, so not in group"}
            group_names = [g.name for g in user.groups]
            if expected["group"] in group_names:
                return {"passed": False, "details": f"User still in {expected['group']}"}
            return {"passed": True, "details": f"User not in {expected['group']}"}

        elif check_type == "group_count":
            user = User.query.filter_by(email=expected["email"]).first()
            if not user:
                return {"passed": False, "details": "User not found"}
            count = len(user.groups)
            if "exact_count" in expected and count != expected["exact_count"]:
                return {"passed": False, "details": f"Group count={count}, expected {expected['exact_count']}"}
            return {"passed": True, "details": f"Group count check passed ({count})"}

        elif check_type == "ticket_status":
            ticket = Ticket.query.get(expected["ticket_id"])
            if not ticket:
                return {"passed": False, "details": f"Ticket #{expected['ticket_id']} not found"}
            if ticket.status != expected["status"]:
                return {"passed": False, "details": f"Ticket status={ticket.status}, expected {expected['status']}"}
            if "notes_contains" in expected and expected["notes_contains"].lower() not in (ticket.notes or "").lower():
                return {"passed": False, "details": f"Notes don't contain '{expected['notes_contains']}'"}
            return {"passed": True, "details": "Ticket status correct"}

        elif check_type == "ticket_status_changed":
            ticket = Ticket.query.get(expected["ticket_id"])
            if not ticket:
                return {"passed": False, "details": "Ticket not found"}
            if ticket.status == "Pending":
                return {"passed": False, "details": "Ticket still Pending — no decision made"}
            return {"passed": True, "details": f"Ticket updated to {ticket.status}"}

        elif check_type == "no_pending_tickets":
            pending = Ticket.query.filter_by(status="Pending").count()
            if pending > 0:
                return {"passed": False, "details": f"{pending} tickets still pending"}
            return {"passed": True, "details": "No pending tickets"}

        elif check_type == "multi":
            all_passed = True
            details = []
            for sub_check in expected["checks"]:
                sub_result = validate_expected(sub_check)
                details.append(sub_result["details"])
                if not sub_result["passed"]:
                    all_passed = False
            return {"passed": all_passed, "details": " | ".join(details)}

        elif check_type in ("error_expected", "read_only", "rag_answer", "custom"):
            # These are validated by the agent's response, not DB state
            return {"passed": True, "details": f"Check type '{check_type}' — pass-through"}

        else:
            return {"passed": False, "details": f"Unknown check type: {check_type}"}


def detect_side_effects(before: dict, after: dict, task: dict) -> list[str]:
    """
    Compare DB snapshots to find unintended changes.
    Returns a list of side-effect descriptions.
    """
    side_effects = []
    forbidden = task.get("forbidden_side_effects")
    if not forbidden:
        return side_effects

    table = forbidden.get("table", "")
    email = forbidden.get("email", "")
    must_not_change = forbidden.get("must_not_change", [])

    if table == "users" and email:
        before_user = next((u for u in before["users"] if u["email"] == email), None)
        after_user = next((u for u in after["users"] if u["email"] == email), None)

        if before_user and after_user:
            for field in must_not_change:
                if before_user.get(field) != after_user.get(field):
                    side_effects.append(
                        f"SIDE EFFECT: {email}.{field} changed from '{before_user.get(field)}' to '{after_user.get(field)}'"
                    )

    return side_effects


# ── Main Harness ─────────────────────────────────────────────────────

CONFIGS = {
    "baseline":    {"rag_mode": "none",   "use_tools": False},
    "tools_only":  {"rag_mode": "none",   "use_tools": True},
    "dense":       {"rag_mode": "dense",  "use_tools": True},
    "bm25":        {"rag_mode": "bm25",   "use_tools": True},
    "hybrid":      {"rag_mode": "hybrid", "use_tools": True},
}


def run_harness(config_name: str = "hybrid", fast: bool = False):
    """Run the evaluation harness."""
    from agent_core import ITAgent

    config = CONFIGS.get(config_name, CONFIGS["hybrid"])
    print(f"\nEVAL HARNESS Config: {config_name}")
    print(f"RAG: {config['rag_mode']} | Tools: {config['use_tools']}")

    # Load tasks
    with open(TASKS_PATH, "r") as f:
        tasks = json.load(f)

    if fast:
        # Smoke test: one task per category
        seen_categories = set()
        subset = []
        for t in tasks:
            if t["category"] not in seen_categories:
                subset.append(t)
                seen_categories.add(t["category"])
        tasks = subset
        print(f"[FAST MODE] Running {len(tasks)} tasks (one per category)\n")

    agent = ITAgent(
        rag_mode=config["rag_mode"],
        use_tools=config["use_tools"],
        use_browser=False,  # Browser agent is too slow for batch eval
    )

    results = []
    total_start = time.time()

    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] {task['id']}: {task['natural_language'][:60]}...")

        # Reset DB to seed state before each task
        reset_db()

        # Snapshot before
        before = snapshot_db()

        # Run the agent
        try:
            result = agent.run(task["natural_language"])
        except Exception as e:
            result_entry = {
                "task_id": task["id"],
                "category": task["category"],
                "difficulty": task["difficulty"],
                "success": False,
                "failure_reason": f"EXCEPTION: {str(e)}",
                "side_effects": [],
                "latency_s": 0,
                "tokens_used": 0,
                "method": "error",
            }
            results.append(result_entry)
            print(f"Exception: {e}")
            continue

        # Snapshot after
        after = snapshot_db()

        # Validate expected DB state
        validation = validate_expected(task["expected_db"])

        # Detect side effects
        side_effects = detect_side_effects(before, after, task)

        # Determine failure reason
        failure_reason = ""
        if not validation["passed"]:
            failure_reason = validation["details"]
        elif not result.success:
            failure_reason = "Agent reported failure"

        task_success = validation["passed"] and result.success

        result_entry = {
            "task_id": task["id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "success": task_success,
            "failure_reason": failure_reason,
            "side_effects": side_effects,
            "latency_s": result.latency_s,
            "tokens_used": result.tokens_used,
            "method": result.method,
            "citations": result.citations,
            "agent_message": result.message[:200] if result.message else "",
        }
        results.append(result_entry)

        status = "[PASS]" if task_success else "[FAIL]"
        se_warning = f" | {len(side_effects)} side effects" if side_effects else ""
        print(f"    {status} {result.latency_s}s, {result.tokens_used} tokens{se_warning}")

    total_elapsed = time.time() - total_start

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(RESULTS_DIR, f"{config_name}_{timestamp}.json")

    output = {
        "config": config_name,
        "config_details": config,
        "total_tasks": len(tasks),
        "total_time_s": round(total_elapsed, 2),
        "timestamp": timestamp,
        "results": results,
    }

    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {result_path}")

    # Print summary
    from metrics import compute_metrics, print_metrics
    metrics = compute_metrics(output)
    print_metrics(metrics, config_name)

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IT Agent Evaluation Harness")
    parser.add_argument("--config", default="hybrid", choices=list(CONFIGS.keys()),
                        help="Agent configuration to evaluate")
    parser.add_argument("--fast", action="store_true",
                        help="Run smoke-test subset (one task per category)")
    args = parser.parse_args()

    run_harness(args.config, args.fast)
