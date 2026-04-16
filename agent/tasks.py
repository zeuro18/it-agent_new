"""
tasks.py
────────
Converts raw natural-language IT requests into detailed, step-by-step
browser task prompts that the browser-use agent can follow.
Instead of passing "reset john's password", we give the agent:
  - the exact URL to start at
  - what fields to fill
  - what success looks like

This keeps the agent reliable across different phrasings of the same task.
"""

import re
import os
PANEL_URL = os.getenv("PANEL_URL", "http://127.0.0.1:5000")

def build_task_prompt(user_input: str) -> str:
    """
    Given a natural-language IT request, return a detailed task prompt
    for the browser-use agent.

    Supports:
      - reset password
      - create / add user
      - delete user
      - assign license
      - check if user exists (conditional)
      - combined: check → create → assign license
    """
    text = user_input.strip().lower()

    # CONDITIONAL: check + create + assign 
    if ("check" in text or "if not" in text or "exist" in text) and \
       ("create" in text or "add" in text) and \
       ("license" in text or "assign" in text):

        email = _extract_email(user_input)
        name  = _extract_name(user_input)
        dept  = _extract_department(user_input)
        sw    = _extract_software(user_input)
        plan  = _extract_plan(user_input)

        return f"""
You are an IT admin assistant. Complete this multi-step task carefully.

TASK: Check if the user {email} exists. If they do NOT exist, create them.
      Then assign a {plan} {sw} license to them.

STEPS:
1. Go to {PANEL_URL}/users
2. Use the search box to search for "{email}"
3. DECISION POINT:
   a. If a user row with "{email}" appears → they already exist, skip to step 5.
   b. If no results appear → proceed to step 4.

4. CREATE THE USER:
   - Click the "Add User" button
   - Fill in Full Name: "{name or email.split('@')[0].title()}"
   - Fill in Email: "{email}"
   - Fill in Department: "{dept or 'General'}"
   - Select Role: "employee"
   - Click "Create User"
   - Wait for the success message confirming the user was created.

5. ASSIGN THE LICENSE:
   - Go to {PANEL_URL}/licenses
   - In the "Assign New License" form:
     * Select the user "{email}" from the User dropdown
     * Select Software: "{sw or 'Microsoft 365'}"
     * Select Plan: "{plan or 'Pro'}"
   - Click "Assign License"
   - Wait for the success message.

6. Verify the license appears in the license table below the form.

SUCCESS: Report whether the user already existed or was newly created,
         and confirm the license was assigned.
""".strip()

    #  RESET PASSWORD 
    if "reset" in text and "password" in text:
        email    = _extract_email(user_input)
        password = _extract_password(user_input)

        return f"""
You are an IT admin assistant. Complete this task:

TASK: Reset the password for {email}.

STEPS:
1. Go to {PANEL_URL}/reset-password
2. In the "User Email" field, type: {email}
3. In the "New Password" field, type: {password or 'Temp@1234'}
4. Click the "Reset Password" button
5. Wait for the green success message to appear on the page.

SUCCESS: Confirm you can see the success message saying the password was reset.
""".strip()

    # CREATE / ADD USER 
    if any(w in text for w in ["create user", "add user", "new user", "create a user", "add a user"]):
        email = _extract_email(user_input)
        name  = _extract_name(user_input)
        dept  = _extract_department(user_input)
        role  = _extract_role(user_input)

        return f"""
You are an IT admin assistant. Complete this task:

TASK: Create a new user account.

STEPS:
1. Go to {PANEL_URL}/users/add
2. Fill in Full Name: "{name or 'New User'}"
3. Fill in Email: "{email}"
4. Fill in Department: "{dept or 'General'}"
5. Select Role: "{role or 'employee'}"
6. Click the "Create User" button
7. Wait for the green success message.

SUCCESS: Confirm the user {email} appears in the users list.
""".strip()

    # DELETE USER 
    if any(w in text for w in ["delete user", "remove user", "deactivate"]):
        email = _extract_email(user_input)

        return f"""
You are an IT admin assistant. Complete this task:

TASK: Delete the user {email}.

STEPS:
1. Go to {PANEL_URL}/users
2. Search for "{email}" in the search box
3. Find the row containing "{email}"
4. Click the red "Delete" button in that row
5. Confirm the deletion in the popup dialog
6. Wait for the success message

SUCCESS: Confirm {email} no longer appears in the users list.
""".strip()

    #ASSIGN LICENSE
    if "assign" in text and "license" in text:
        email = _extract_email(user_input)
        sw    = _extract_software(user_input)
        plan  = _extract_plan(user_input)

        return f"""
You are an IT admin assistant. Complete this task:

TASK: Assign a software license to {email}.

STEPS:
1. Go to {PANEL_URL}/licenses
2. In the "Assign New License" form:
   - Select the user "{email}" from the User dropdown
   - Select Software: "{sw or 'Microsoft 365'}"
   - Select Plan: "{plan or 'Pro'}"
3. Click "Assign License"
4. Wait for the success message

SUCCESS: Confirm the license appears in the license assignments table.
""".strip()

    #REVOKE LICENSE 
    if "revoke" in text and "license" in text:
        email = _extract_email(user_input)
        sw    = _extract_software(user_input)

        return f"""
You are an IT admin assistant. Complete this task:

TASK: Revoke the {sw or 'software'} license from {email}.

STEPS:
1. Go to {PANEL_URL}/licenses
2. Find the row where Email is "{email}" and Software is "{sw or ''}"
3. Click the "Revoke" button in that row
4. Confirm the action in the popup dialog
5. Wait for the success message

SUCCESS: Confirm the license no longer appears in the table.
""".strip()

    #CHECK USER EXISTS 
    if "check" in text and ("exist" in text or "user" in text):
        email = _extract_email(user_input)

        return f"""
You are an IT admin assistant. Complete this task:

TASK: Check whether the user {email} exists in the system.

STEPS:
1. Go to {PANEL_URL}/users
2. Search for "{email}" in the search box
3. Report whether a user with that email was found or not

SUCCESS: Clearly state "User EXISTS" or "User NOT FOUND" based on what you see.
""".strip()

    #FALLBACK: pass through as-is 
    return f"""
You are an IT admin assistant working on the admin panel at {PANEL_URL}.

TASK: {user_input}

Navigate the admin panel to complete this task. The panel has these pages:
- {PANEL_URL}/          → Dashboard
- {PANEL_URL}/users     → List / search / add / edit / delete users
- {PANEL_URL}/users/add → Add new user form
- {PANEL_URL}/reset-password → Reset a user's password
- {PANEL_URL}/licenses  → Assign / revoke software licenses
- {PANEL_URL}/audit     → View audit log

Complete the task and confirm success.
""".strip()


# HELPER EXTRACTORS 

def _extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "user@company.com"

def _extract_password(text: str) -> str:
    # Matches patterns like "to Temp@1234" or "password: Temp@1234"
    match = re.search(r"(?:to|password[:\s]+|new password[:\s]+)\s*([^\s,]+)", text, re.IGNORECASE)
    if match:
        candidate = match.group(1)
        # Make sure it's not another keyword
        if candidate.lower() not in ("temp", "the", "a", "be", "is", "for"):
            return candidate
    return "Temp@1234"

def _extract_name(text: str) -> str:
    match = re.search(r"(?:named?|for|name[:\s]+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
    return match.group(1) if match else ""

def _extract_department(text: str) -> str:
    match = re.search(r"(?:in|department[:\s]+|dept[:\s]+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
    if match:
        val = match.group(1)
        if val.lower() not in ("the", "a", "an"):
            return val
    return ""

def _extract_role(text: str) -> str:
    for role in ["admin", "manager", "employee"]:
        if role in text.lower():
            return role
    return "employee"

def _extract_software(text: str) -> str:
    software_map = {
        "microsoft 365": "Microsoft 365",
        "office 365": "Microsoft 365",
        "m365": "Microsoft 365",
        "slack": "Slack",
        "zoom": "Zoom",
        "github": "GitHub",
        "jira": "Jira",
        "figma": "Figma",
    }
    lower = text.lower()
    for key, val in software_map.items():
        if key in lower:
            return val
    return "Microsoft 365"

def _extract_plan(text: str) -> str:
    for plan in ["enterprise", "business", "pro", "standard"]:
        if plan in text.lower():
            return plan.title()
    return "Pro"
