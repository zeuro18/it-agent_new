import sys
import os
from datetime import datetime

# Ensure we can import from the root directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from app import app
from database import db, User, License, AuditLog, Group, Ticket

def get_app_context():
    return app.app_context()

def user_lookup(email: str) -> dict | None:
    """Find a user by email and return their details."""
    with get_app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return None
        return {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role,
            "department": user.department,
            "status": user.status,
            "groups": [g.name for g in user.groups]
        }

def list_licenses(email: str) -> list[dict]:
    """Get all software licenses assigned to a user."""
    with get_app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return []
        licenses = License.query.filter_by(assigned_to=user.id).all()
        return [{"software": l.software, "plan": l.plan, "date": str(l.assigned_date)} for l in licenses]

def assign_license(email: str, software: str, plan: str) -> dict:
    """Assign a license to a user."""
    with get_app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return {"success": False, "error": f"User {email} not found"}
        
        existing = License.query.filter_by(assigned_to=user.id, software=software).first()
        if existing:
            return {"success": False, "error": f"User already has {software}"}
            
        lic = License(software=software, assigned_to=user.id, plan=plan, assigned_date=datetime.utcnow())
        db.session.add(lic)
        log = AuditLog(action=f"Assigned {software} ({plan}) to {user.email}", performed_by="Tool Agent")
        db.session.add(log)
        db.session.commit()
        return {"success": True, "message": f"Assigned {software} to {email}"}

def revoke_license(email: str, software: str) -> dict:
    """Revoke a license from a user."""
    with get_app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return {"success": False, "error": f"User {email} not found"}
            
        lic = License.query.filter_by(assigned_to=user.id, software=software).first()
        if not lic:
            return {"success": False, "error": f"User does not have {software}"}
            
        db.session.delete(lic)
        log = AuditLog(action=f"Revoked {software} from {email}", performed_by="Tool Agent")
        db.session.add(log)
        db.session.commit()
        return {"success": True, "message": f"Revoked {software} from {email}"}

def list_pending_tickets() -> list[dict]:
    """List all pending tickets that require IT review."""
    with get_app_context():
        tickets = Ticket.query.filter_by(status="Pending").all()
        result = []
        for t in tickets:
            user = User.query.get(t.created_for)
            result.append({
                "ticket_id": t.id,
                "email": user.email if user else "Unknown",
                "issue": t.issue,
                "priority": t.priority,
                "notes": t.notes
            })
        return result

def update_ticket(ticket_id: int, status: str, notes: str = "") -> dict:
    """Update ticket status (e.g., 'Approved', 'Rejected') and add notes."""
    with get_app_context():
        ticket = Ticket.query.get(ticket_id)
        if not ticket:
            return {"success": False, "error": f"Ticket #{ticket_id} not found"}
        
        ticket.status = status
        if notes:
            ticket.notes = notes
            
        log = AuditLog(action=f"Updated ticket #{ticket_id} to {status}", performed_by="Tool Agent")
        db.session.add(log)
        db.session.commit()
        return {"success": True, "message": f"Ticket #{ticket_id} marked as {status}"}

def assign_user_to_group(email: str, group_name: str) -> dict:
    """Add a user to a group."""
    with get_app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return {"success": False, "error": "User not found"}
            
        group = Group.query.filter_by(name=group_name).first()
        if not group:
            return {"success": False, "error": "Group not found"}
            
        if group in user.groups:
            return {"success": False, "error": "User already in group"}
            
        user.groups.append(group)
        log = AuditLog(action=f"Assigned user {email} to group {group_name}", performed_by="Tool Agent")
        db.session.add(log)
        db.session.commit()
        return {"success": True, "message": f"Assigned {email} to {group_name}"}

def remove_user_from_group(email: str, group_name: str) -> dict:
    """Remove a user from a group."""
    with get_app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            return {"success": False, "error": "User not found"}
            
        group = Group.query.filter_by(name=group_name).first()
        if not group:
            return {"success": False, "error": "Group not found"}
            
        if group not in user.groups:
            return {"success": False, "error": "User not in group"}
            
        user.groups.remove(group)
        log = AuditLog(action=f"Removed user {email} from group {group_name}", performed_by="Tool Agent")
        db.session.add(log)
        db.session.commit()
        return {"success": True, "message": f"Removed {email} from {group_name}"}

# List of available tool callables for the agent
AVAILABLE_TOOLS = [
    user_lookup, list_licenses, assign_license, revoke_license,
    list_pending_tickets, update_ticket, assign_user_to_group, remove_user_from_group
]
