from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
db = SQLAlchemy()

user_group = db.Table('user_group',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('group_id', db.Integer, db.ForeignKey('groups.id'), primary_key=True)
)

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    role = db.Column(db.String(50), default="employee")
    department = db.Column(db.String(100), default="General")
    status = db.Column(db.String(20), default="active") 
    password_hint = db.Column(db.String(100), default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    groups = db.relationship('Group', secondary=user_group, backref=db.backref('users', lazy='dynamic'))

class Group(db.Model):
    __tablename__ = "groups"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Ticket(db.Model):
    __tablename__ = "tickets"
    id = db.Column(db.Integer, primary_key=True)
    created_for = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    issue = db.Column(db.String(300), nullable=False)
    priority = db.Column(db.String(20), default="Medium")
    status = db.Column(db.String(20), default="Pending") # Pending, Approved, Rejected, Resolved
    notes = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class License(db.Model):
    __tablename__ = "licenses"
    id = db.Column(db.Integer, primary_key=True)
    software = db.Column(db.String(100), nullable=False)
    assigned_to = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    plan = db.Column(db.String(50), default="Standard")
    assigned_date = db.Column(db.DateTime, default=datetime.utcnow)

class AuditLog(db.Model):
    __tablename__ = "audit_logs"
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(300), nullable=False)
    performed_by = db.Column(db.String(100), default="Admin")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


def seed_db():
    """Populate an empty database with the fixed demo dataset.

    The eval harness resets to this state before every task, so the seed
    must stay deterministic: the insert order fixes the autoincrement IDs
    that tasks and tools rely on (users 1-5, licenses 1-3, groups 1-4,
    tickets 1-3).
    """
    users = [
        User(name="John Doe", email="john@company.com", role="employee", status="active", department="Engineering"),
        User(name="Sarah Connor", email="sarah@company.com", role="manager", status="active", department="HR"),
        User(name="Marcus Vance", email="marcus@company.com", role="employee", status="inactive", department="Sales"),
        User(name="Daniel Craig", email="daniel@company.com", role="employee", status="inactive", department="IT"),
        User(name="Emily Watson", email="Emily@company.com", role="employee", status="active", department="Legal")
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

    users[0].groups.append(groups[0])
    users[1].groups.append(groups[2])
    db.session.commit()

    tickets = [
        Ticket(created_for=1, issue="Requesting access to GitHub Copilot", priority="Medium", status="Pending"),
        Ticket(created_for=2, issue="Need a new laptop", priority="High", status="Approved", notes="Processing order"),
        Ticket(created_for=3, issue="Cannot access Jira", priority="High", status="Resolved", notes="Reset password")
    ]
    db.session.add_all(tickets)
    db.session.commit()
