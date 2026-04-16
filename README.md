
## Architecture

The project is structurally divided into two interacting components:

1. The Admin Panel 
   A Flask-based web application that mimics a standard internal IT department portal. It exposes various user interfaces (Dashboards, User Management tables, License Assignment forms) built with Jinja2 templates, and interacts with a local SQLite database to persist state and audit logs.

2. The IT Agent (Automated Operator): 
   A Python Command-Line Interface (CLI) that leverages the `browser-use` library and LangChain. 
   - A user provides a request in plain English.
   - The `tasks.py` module parses the request and expands it into a deterministic, step-by-step instruction prompt (including the target URLs and exact goals).
   - The Agent initiates a Playwright-controlled Chromium browser instance and enters a text-based DOM processing loop: it parses the HTML DOM tree as text elements, sends it to the language model (LLM), and receives the next browser action (click, type, navigate) via JSON function calling. 
   - It continually executes actions on the Flask admin panel until the system state confirms the IT task has succeeded.

---

## Tech Stack

### Web Application / Backend
- **Framework:** Flask (v3.0.3)
- **Database ORM:** Flask-SQLAlchemy (v3.1.1)
- **Database:** SQLite (`itadmin.db` - local lightweight database)
- **Frontend Template Engine:** Jinja2
- **Styling/Markup:** Vanilla HTML & CSS

### AI Agent / Browser Automation
- **Framework:** `browser-use` (An AI framework orchestrating Playwright in text-only mode)
- **Browser Automation:** Playwright for Python
- **LLM/Orchestration:** LangChain (`langchain_groq`)
- **Language Model:** Groq API running `llama-3.3-70b-versatile`

---

## Features

- **Natural Language Command Processing:** Say "reset password for alice@company.com to New123" and the system takes care of navigating the UI.
- **Complex Conditional Workflows:** Supports advanced chained reasoning directly in the browser. Example: "Check if bob@company.com exists; if not, create him and assign a Pro Slack license."
- **User Management Operations:** View, Search, Create, Edit, and Delete employees. 
- **Software License Provisioning:** Readily assign or revoke software licenses (Microsoft 365, Slack, GitHub, Jira, etc.) to individuals based on available plans.
- **Persistent Audit Logging:** Every manipulation of users or licenses made via the Admin dashboard is recorded in a tamper-proof audit log table.
- **Execution Modes:**
  - **Interactive CLI:** Type `python main.py` to enter an interactive conversation terminal with the AI agent.
  - **Single Task Execution:** Provide explicit commands upon initialization (e.g., `python main.py "delete user x@company.com"`).

