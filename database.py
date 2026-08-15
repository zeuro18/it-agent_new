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
