"""
harness.py
Evaluation harness for the IT Agent.

Runs tasks from tasks_bank.json against the agent, validates DB state
against ground truth, detects side effects, and records metrics.

Usage:
    python eval/harness.py                          # Run all tasks with default config
    python eval/harness.py --config baseline        # No tools, no RAG (browser only concept)
    python eval/harness.py --config tools_only      # Tools, no RAG
    python eval/harness.py --config dense           # Tools + Dense RAG
    python eval/harness.py --config bm25            # Tools + BM25 RAG
    python eval/harness.py --config hybrid          # Tools + Hybrid RAG
    python eval/harness.py --fast                   # Run smoke-test subset only
    python eval/harness.py --category injection     # Run one category only
    python eval/harness.py --no-guardrails          # Disable prompt guardrails
"""

import os
import re
import sys
import json
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'agent'))

from app import app
from database import db, User, License, AuditLog, Group, Ticket, seed_db

TASKS_PATH = os.path.join(os.path.dirname(__file__), "tasks_bank.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
# Proactive gap between tasks within a single harness run (seconds). Override
# with EVAL_TASK_PACING_S=0 to disable, or raise it if you're still seeing
# reactive 429 waits in agent_core._invoke.
TASK_PACING_S = float(os.getenv("EVAL_TASK_PACING_S", "2.0"))


# DB snapshot and reset

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
        seed_db()


# Task setup

def apply_setup(setup: dict):
    """
    Apply a task's preconditions to the DB before the agent runs.

    Supported keys:
      - ensure_user_exists: <email>
          create a default user if missing
      - ensure_user_not_exists: <email>
          remove the user (and dependents) if present
      - ensure_license: {email, software, plan}
          grant the license if the user exists and lacks it
      - ensure_group_member: {email, group}
          add the user to the group if both exist and they're not a member

    Seeded users (see seed_db) already satisfy most ensure_user_exists
    preconditions; this only creates a user when the task needs one that
    isn't part of the fixed seed set (e.g. daniel@company.com in tasks that
    reuse it, or any future ad-hoc email).
    """
    if not setup:
        return

    with app.app_context():
        if "ensure_user_exists" in setup:
            email = setup["ensure_user_exists"]
            user = User.query.filter_by(email=email).first()
            if not user:
                local = email.split("@")[0]
                name = local.replace(".", " ").replace("_", " ").title()
                user = User(name=name, email=email, role="employee",
                            status="active", department="General")
                db.session.add(user)
                db.session.commit()

        if "ensure_user_not_exists" in setup:
            email = setup["ensure_user_not_exists"]
            user = User.query.filter_by(email=email).first()
            if user:
                License.query.filter_by(assigned_to=user.id).delete()
                Ticket.query.filter_by(created_for=user.id).delete()
                user.groups.clear()
                db.session.delete(user)
                db.session.commit()

        if "ensure_license" in setup:
            spec = setup["ensure_license"]
            user = User.query.filter_by(email=spec["email"]).first()
            if user and not License.query.filter_by(assigned_to=user.id, software=spec["software"]).first():
                db.session.add(License(software=spec["software"], assigned_to=user.id,
                                       plan=spec.get("plan", "Standard"),
                                       assigned_date=datetime.utcnow()))
                db.session.commit()

        if "ensure_group_member" in setup:
            spec = setup["ensure_group_member"]
            user = User.query.filter_by(email=spec["email"]).first()
            group = Group.query.filter_by(name=spec["group"]).first()
            if user and group and group not in user.groups:
                user.groups.append(group)
                db.session.commit()

        if "inject_ticket_note" in setup:
            spec = setup["inject_ticket_note"]
            ticket = db.session.get(Ticket, spec["ticket_id"])
            if ticket:
                ticket.notes = (ticket.notes + "\n" if ticket.notes else "") + spec["note"]
                db.session.commit()
        # pre_authorized is metadata for the harness/policy; no DB action needed.


# Validation checks

_NOT_FOUND_VARIANTS = [
    "not found", "does not exist", "doesn't exist", "do not exist",
    "don't exist", "no results", "nonexistent", "no such", "unknown",
]


def _canonicalize_not_found(text: str) -> str:
    """Rewrite common 'resource is missing' phrasings to a single token so
    error_expected matching isn't tied to one wording."""
    for variant in _NOT_FOUND_VARIANTS:
        text = text.replace(variant, " notfound ")
    return re.sub(r"\s+", " ", text)


def validate_expected(expected: dict, agent_message: str = "") -> dict:
    """
    Validate the expected DB state (or agent response, for soft checks)
    after a task.
    Returns {"passed": bool, "details": str}
    """
    check_type = expected.get("check")
    msg_lower = (agent_message or "").lower()

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
            if "equals" in expected and (val or "") != expected["equals"]:
                return {"passed": False, "details": f"Field {expected['field']} = '{val}', expected '{expected['equals']}'"}
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
            ticket = db.session.get(Ticket, expected["ticket_id"])
            if not ticket:
                return {"passed": False, "details": f"Ticket #{expected['ticket_id']} not found"}
            if ticket.status != expected["status"]:
                return {"passed": False, "details": f"Ticket status={ticket.status}, expected {expected['status']}"}
            if "notes_contains" in expected and expected["notes_contains"].lower() not in (ticket.notes or "").lower():
                return {"passed": False, "details": f"Notes don't contain '{expected['notes_contains']}'"}
            return {"passed": True, "details": "Ticket status correct"}

        elif check_type == "ticket_status_changed":
            ticket = db.session.get(Ticket, expected["ticket_id"])
            if not ticket:
                return {"passed": False, "details": "Ticket not found"}
            if ticket.status == "Pending":
                return {"passed": False, "details": "Ticket still Pending, no decision made"}
            return {"passed": True, "details": f"Ticket updated to {ticket.status}"}

        elif check_type == "group_no_license":
            group = Group.query.filter_by(name=expected["group"]).first()
            if not group:
                return {"passed": False, "details": f"Group {expected['group']} not found"}
            offenders = []
            for u in group.users:
                if License.query.filter_by(assigned_to=u.id, software=expected["software"]).first():
                    offenders.append(u.email)
            if offenders:
                return {"passed": False,
                        "details": f"Users still hold {expected['software']}: {', '.join(offenders)}"}
            return {"passed": True, "details": f"No {expected['group']} member holds {expected['software']}"}

        elif check_type == "no_pending_tickets":
            pending = Ticket.query.filter_by(status="Pending").count()
            if pending > 0:
                return {"passed": False, "details": f"{pending} tickets still pending"}
            return {"passed": True, "details": "No pending tickets"}

        elif check_type == "multi":
            all_passed = True
            details = []
            for sub_check in expected["checks"]:
                sub_result = validate_expected(sub_check, agent_message)
                details.append(sub_result["details"])
                if not sub_result["passed"]:
                    all_passed = False
            return {"passed": all_passed, "details": " | ".join(details)}

        elif check_type == "rag_answer":
            must_mention = expected.get("must_mention", [])
            if not must_mention:
                return {"passed": True, "details": "No must_mention terms specified"}
            hits = [term for term in must_mention if term.lower() in msg_lower]
            if not hits:
                return {"passed": False,
                        "details": f"Agent message mentioned none of {must_mention}"}
            return {"passed": True, "details": f"Agent message mentioned: {hits}"}

        elif check_type == "error_expected":
            reason = expected.get("reason", "")
            if not reason:
                return {"passed": True, "details": "No reason specified"}
            if reason.lower() in msg_lower:
                return {"passed": True,
                        "details": "Agent message contains the expected failure text"}
            # The agent may phrase the same failure differently than the
            # ground-truth reason string ("not found" vs "does not exist" vs
            # "no results"), so canonicalize those variants to one token
            # before falling back to keyword overlap. Email addresses are
            # stripped from the reason so their domain words don't become
            # required keywords.
            canonical_msg = _canonicalize_not_found(msg_lower)
            reason_wo_emails = re.sub(r"\S+@\S+", " ", reason.lower())
            canonical_reason = _canonicalize_not_found(reason_wo_emails)
            if canonical_reason in canonical_msg:
                return {"passed": True,
                        "details": "Agent message contains the expected failure text"}
            stopwords = {"the", "a", "an", "is", "are", "of", "to", "for", "with"}
            keywords = [w for w in re.findall(r"[a-z]+", canonical_reason)
                        if w not in stopwords and len(w) > 2]
            if keywords and all(k in canonical_msg for k in keywords):
                return {"passed": True,
                        "details": f"Agent message covers expected failure keywords {keywords}"}
            return {"passed": False,
                    "details": f"Agent message does not reflect expected failure: '{reason}'"}

        elif check_type in ("read_only", "custom"):
            # These aren't verifiable from DB state alone; they're covered
            # by side-effect detection (read_only) or need manual review (custom).
            return {"passed": True, "details": f"Check type '{check_type}' is a pass-through"}

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


def diff_snapshots(before: dict, after: dict) -> list[str]:
    """Describe every difference between two snapshots (used for Q&A tasks,
    which must not mutate any state, including the audit log)."""
    changes = []
    for table in ("users", "licenses", "tickets", "groups"):
        if before[table] != after[table]:
            changes.append(f"{table} table changed")
    if before["audit_count"] != after["audit_count"]:
        delta = after["audit_count"] - before["audit_count"]
        changes.append(f"audit log grew by {delta}")
    return changes


# Main harness

CONFIGS = {
    "baseline":          {"rag_mode": "none",   "use_tools": False, "verify_mode": "tool",  "use_policy": False},
    "tools_only":        {"rag_mode": "none",   "use_tools": True,  "verify_mode": "tool",  "use_policy": False},
    "dense":             {"rag_mode": "dense",  "use_tools": True,  "verify_mode": "tool",  "use_policy": False},
    "bm25":              {"rag_mode": "bm25",   "use_tools": True,  "verify_mode": "tool",  "use_policy": False},
    "hybrid":            {"rag_mode": "hybrid", "use_tools": True,  "verify_mode": "tool",  "use_policy": False},
    # Experiment A: DB-state verification
    "hybrid_db_verify":  {"rag_mode": "hybrid", "use_tools": True,  "verify_mode": "db",    "use_policy": False},
    # Experiment B: Deterministic policy engine
    "hybrid_no_policy":  {"rag_mode": "hybrid", "use_tools": True,  "verify_mode": "tool",  "use_policy": False},
    "hybrid_policy":     {"rag_mode": "hybrid", "use_tools": True,  "verify_mode": "tool",  "use_policy": True},
}


import random

_FLAKY_PATCHED = False
_FLAKY_ACTIVE = False

def _patch_flaky_writes(rate: float = 0.15):
    """
    Simulate flaky database writes (for Phase 1 Experiment A).
    During agent tool execution, randomly drops ~15% of DB commits
    by rolling back instead, testing whether verifier.py catches the mismatch.
    """
    global _FLAKY_PATCHED, _FLAKY_ACTIVE
    _FLAKY_ACTIVE = True
    if _FLAKY_PATCHED:
        return
    _FLAKY_PATCHED = True

    orig_commit = db.session.commit

    def flaky_commit():
        if _FLAKY_ACTIVE and random.random() < rate:
            db.session.rollback()
            print("    [FLAKY WRITE SIMULATED] Dropped DB commit!")
            return
        return orig_commit()

    db.session.commit = flaky_commit


def run_harness(config_name: str = "hybrid", fast: bool = False, use_browser: bool = False,
                guardrails: bool = True, category: str = None,
                simulate_flaky_writes: bool = False):
    """Run the evaluation harness."""
    from agent_core import ITAgent

    config = CONFIGS.get(config_name, CONFIGS["hybrid"])
    print(f"\nEVAL HARNESS Config: {config_name}")
    print(f"RAG: {config['rag_mode']} | Tools: {config['use_tools']} | "
          f"Browser fallback: {use_browser} | Guardrails: {guardrails} | "
          f"VerifyMode: {config.get('verify_mode', 'tool')} | "
          f"Policy: {config.get('use_policy', False)}"
          + (f" | FlakySim: ON" if simulate_flaky_writes else "")
          + (f" | Category: {category}" if category else ""))

    if simulate_flaky_writes:
        _patch_flaky_writes()

    if use_browser:
        # Eval runs shouldn't pop up a visible browser window.
        os.environ.setdefault("BROWSER_HEADLESS", "1")

    # Load tasks
    with open(TASKS_PATH, "r") as f:
        tasks = json.load(f)

    if category:
        tasks = [t for t in tasks if t["category"] == category]
        if not tasks:
            print(f"No tasks in category '{category}'")
            return

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
        use_browser=use_browser,
        guardrails=guardrails,
        verify_mode=config.get("verify_mode", "tool"),
        use_policy=config.get("use_policy", False),
    )

    results = []
    total_start = time.time()

    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] {task['id']}: {task['natural_language'][:60]}...", flush=True)

        # Fresh seed state, then task-specific preconditions, before each run
        reset_db()
        apply_setup(task.get("setup"))

        # Extract pre_authorized flag from setup for the policy engine
        pre_authorized = bool((task.get("setup") or {}).get("pre_authorized", False))
        agent.pre_authorized = pre_authorized

        before = snapshot_db()

        try:
            result = agent.run(task["natural_language"], task_id=task["id"])
            if result.trace:
                traces_dir = os.path.join(RESULTS_DIR, "traces")
                result.trace.save(traces_dir)
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

        after = snapshot_db()

        # Validate expected DB state (or agent response, for soft checks)
        validation = validate_expected(task["expected_db"], result.message)

        # Citation check
        citations_ok = None
        expected_sources = task["expected_db"].get("must_cite")
        if expected_sources:
            cited_text = " ".join(result.citations or [])
            citations_ok = any(src in cited_text for src in expected_sources)
            if not citations_ok:
                detail = f"Expected citation to {expected_sources}, got {result.citations}"
                validation = {
                    "passed": False,
                    "details": f"{validation['details']} | {detail}" if not validation["passed"] else detail,
                }

        # Detect side effects. Q&A tasks (rag_answer) must not mutate any
        # state at all, so any snapshot difference, audit rows included,
        # counts as a side effect.
        side_effects = detect_side_effects(before, after, task)
        check_type = task["expected_db"].get("check")
        if check_type == "rag_answer":
            side_effects += [f"Q&A task mutated state: {c}" for c in diff_snapshots(before, after)]

        # Categories where tools aren't expected to run at all (pure
        # policy/read-only Q&A) or where the "correct" tool outcome is a
        # reported business-rule failure (error_expected): for these,
        # result.success (which reflects tool-call success/failure) isn't
        # a meaningful signal, so we judge purely on validate_expected.
        gate_on_tool_success = check_type not in ("rag_answer", "read_only", "error_expected")

        # Determine failure reason
        failure_reason = ""
        if not validation["passed"]:
            failure_reason = validation["details"]
        elif gate_on_tool_success and not result.success:
            failure_reason = "Agent reported failure"

        task_success = validation["passed"] and (result.success if gate_on_tool_success else True)
        silent_failure = result.success and not validation["passed"]

        # Count policy-denied calls (Experiment B metric)
        policy_denied = sum(
            1 for e in (result.evidence or [])
            if isinstance(e.get("result"), dict)
            and "Policy denied" in e["result"].get("error", "")
        )

        result_entry = {
            "task_id": task["id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "success": task_success,
            "silent_failure": silent_failure,
            "failure_reason": failure_reason,
            "side_effects": side_effects,
            "latency_s": result.latency_s,
            "tokens_used": result.tokens_used,
            "method": result.method,
            "citations": result.citations,
            "citations_ok": citations_ok,
            "repaired": result.repaired,
            "agent_message": result.message[:200] if result.message else "",
            "pre_authorized": pre_authorized,
            "policy_denied": policy_denied,
        }
        results.append(result_entry)

        status = "[PASS]" if task_success else "[FAIL]"
        se_warning = f" | {len(side_effects)} side effects" if side_effects else ""
        print(f"    {status} {result.latency_s}s, {result.tokens_used} tokens{se_warning}", flush=True)

        # Small proactive gap between tasks. _invoke() already retries
        # reactively after a 429, but that can sleep up to 5 minutes per hit;
        # spreading requests out here means fewer of those reactive waits in
        # the first place. Cheap insurance, not a real rate limiter.
        if i < len(tasks) - 1:
            time.sleep(TASK_PACING_S)

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
    parser.add_argument("--use-browser", action="store_true",
                        help="Enable the browser-agent fallback (slow; drives a real Playwright browser)")
    parser.add_argument("--no-guardrails", action="store_true",
                        help="Disable prompt guardrails (for injection red-team before/after runs)")
    parser.add_argument("--category", help="Run only tasks from this category")
    parser.add_argument("--simulate-flaky-writes", action="store_true",
                        help="Randomly drop ~10%% of DB commits to test verifier.py catches them")
    args = parser.parse_args()

    run_harness(args.config, args.fast, args.use_browser,
                guardrails=not args.no_guardrails, category=args.category,
                simulate_flaky_writes=args.simulate_flaky_writes)