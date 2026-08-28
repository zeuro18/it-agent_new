from flask import Flask, render_template, request, redirect, url_for, flash
from database import db, User, License, AuditLog, Group, Ticket, seed_db
from datetime import datetime
import os
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///itadmin.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.create_all()
    if User.query.count() == 0:
        seed_db()

# DASHBOARD

@app.route("/")
def dashboard():
    total_users = User.query.count()
    active_users = User.query.filter_by(status="active").count()
    total_licenses = License.query.count()
    recent_logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(5).all()
    return render_template("dashboard.html",
        total_users=total_users,
        active_users=active_users,
        total_licenses=total_licenses,
        recent_logs=recent_logs)

# USERS

@app.route("/users")
def users():
    search = request.args.get("search", "").strip()
    if search:
        all_users = User.query.filter((User.name.ilike(f"%{search}%")) | (User.email.ilike(f"%{search}%"))).all()
    else:
        all_users = User.query.all()
    return render_template("users.html", users=all_users, search=search)


@app.route("/users/add", methods=["GET", "POST"])
def add_user():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip()
        role = request.form["role"]
        department = request.form["department"].strip()

        if User.query.filter_by(email=email).first():
            flash(f"User with email {email} already exists.", "error")
            return render_template("add_user.html", form=request.form)

        user = User(name=name, email=email, role=role, department=department, status="active")
        db.session.add(user)
        db.session.commit()
        log = AuditLog(action=f"Created user {email}", performed_by="IT Agent")
        db.session.add(log)
        db.session.commit()
        flash(f"User {name} ({email}) created successfully!", "success")
        return redirect(url_for("users"))

    return render_template("add_user.html", form={})


@app.route("/users/<int:user_id>/edit", methods=["GET", "POST"])
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    if request.method == "POST":
        user.name = request.form["name"].strip()
        user.role = request.form["role"]
        user.department = request.form["department"].strip()
        user.status = request.form["status"]
        db.session.commit()
        log = AuditLog(action=f"Edited user {user.email}", performed_by="IT Agent")
        db.session.add(log)
        db.session.commit()

        flash(f"User {user.name} updated successfully!", "success")
        return redirect(url_for("users"))

    return render_template("edit_user.html", user=user)


@app.route("/users/<int:user_id>/delete", methods=["POST"])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    email = user.email
    License.query.filter_by(assigned_to=user_id).delete()
    Ticket.query.filter_by(created_for=user_id).delete()
    db.session.delete(user)
    log = AuditLog(action=f"Deleted user {email}", performed_by="IT Agent")
    db.session.add(log)
    db.session.commit()
    flash(f"User {email} deleted.", "success")
    return redirect(url_for("users"))

# RESET PASSWORD

@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        email = request.form["email"].strip()
        new_password = request.form["new_password"].strip()

        user = User.query.filter_by(email=email).first()
        if not user:
            flash(f"No user found with email {email}.", "error")
            return render_template("reset_password.html", form=request.form)

        user.password_hint = f"[reset on {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}]"
        db.session.commit()

        log = AuditLog(action=f"Reset password for {email}", performed_by="IT Agent")
        db.session.add(log)
        db.session.commit()

        flash(f"Password for {email} has been reset successfully!", "success")
        return redirect(url_for("reset_password"))

    return render_template("reset_password.html", form={})

# LICENSES

@app.route("/licenses")
def licenses():
    all_licenses = db.session.query(License, User).join(User, License.assigned_to == User.id).all()
    all_users = User.query.filter_by(status="active").all()
    return render_template("licenses.html", licenses=all_licenses, users=all_users)


@app.route("/licenses/assign", methods=["POST"])
def assign_license():
    user_id = int(request.form["user_id"])
    software = request.form["software"].strip()
    plan = request.form["plan"]
    user = User.query.get_or_404(user_id)

    existing = License.query.filter_by(assigned_to=user_id, software=software).first()
    if existing:
        flash(f"{user.email} already has a {software} license.", "error")
        return redirect(url_for("licenses"))

    lic = License(software=software, assigned_to=user_id, plan=plan, assigned_date=datetime.utcnow())
    db.session.add(lic)
    log = AuditLog(action=f"Assigned {software} ({plan}) to {user.email}", performed_by="IT Agent")
    db.session.add(log)
    db.session.commit()

    flash(f"{software} ({plan}) assigned to {user.email}!", "success")
    return redirect(url_for("licenses"))


@app.route("/licenses/<int:lic_id>/revoke", methods=["POST"])
def revoke_license(lic_id):
    lic = License.query.get_or_404(lic_id)
    user = User.query.get(lic.assigned_to)
    info = f"{lic.software} from {user.email if user else 'unknown'}"
    db.session.delete(lic)

    log = AuditLog(action=f"Revoked {info}", performed_by="IT Agent")
    db.session.add(log)
    db.session.commit()

    flash(f"License revoked: {info}", "success")
    return redirect(url_for("licenses"))

# AUDIT LOG

@app.route("/audit")
def audit():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).all()
    return render_template("audit.html", logs=logs)

# GROUPS

@app.route("/groups")
def groups():
    all_groups = Group.query.all()
    all_users = User.query.filter_by(status="active").all()
    return render_template("groups.html", groups=all_groups, users=all_users)

@app.route("/groups/add", methods=["POST"])
def add_group():
    name = request.form["name"].strip()
    description = request.form["description"].strip()
    if Group.query.filter_by(name=name).first():
        flash(f"Group {name} already exists.", "error")
    else:
        group = Group(name=name, description=description)
        db.session.add(group)
        log = AuditLog(action=f"Created group {name}", performed_by="IT Agent")
        db.session.add(log)
        db.session.commit()
        flash(f"Group {name} created successfully!", "success")
    return redirect(url_for("groups"))

@app.route("/groups/<int:group_id>/assign", methods=["POST"])
def assign_group(group_id):
    user_id = int(request.form["user_id"])
    group = Group.query.get_or_404(group_id)
    user = User.query.get_or_404(user_id)
    if group not in user.groups:
        user.groups.append(group)
        log = AuditLog(action=f"Assigned user {user.email} to group {group.name}", performed_by="IT Agent")
        db.session.add(log)
        db.session.commit()
        flash(f"User {user.email} added to {group.name}", "success")
    else:
        flash(f"User {user.email} is already in {group.name}", "error")
    return redirect(url_for("groups"))

@app.route("/groups/<int:group_id>/remove", methods=["POST"])
def remove_group(group_id):
    user_id = int(request.form["user_id"])
    group = Group.query.get_or_404(group_id)
    user = User.query.get_or_404(user_id)
    if group in user.groups:
        user.groups.remove(group)
        log = AuditLog(action=f"Removed user {user.email} from group {group.name}", performed_by="IT Agent")
        db.session.add(log)
        db.session.commit()
        flash(f"User {user.email} removed from {group.name}", "success")
    return redirect(url_for("groups"))

# TICKETS

@app.route("/tickets")
def tickets():
    all_tickets = db.session.query(Ticket, User).join(User, Ticket.created_for == User.id).all()
    return render_template("tickets.html", tickets=all_tickets)

@app.route("/tickets/<int:ticket_id>/update", methods=["POST"])
def update_ticket(ticket_id):
    ticket = Ticket.query.get_or_404(ticket_id)
    status = request.form["status"]
    notes = request.form["notes"].strip()
    ticket.status = status
    if notes:
        ticket.notes = notes
    log = AuditLog(action=f"Updated ticket #{ticket.id} to {status}", performed_by="IT Agent")
    db.session.add(log)
    db.session.commit()
    flash(f"Ticket #{ticket.id} updated to {status}.", "success")
    return redirect(url_for("tickets"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
