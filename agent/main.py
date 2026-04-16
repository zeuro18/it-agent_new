"""
main.py
───────
CLI entry point for the IT Agent.

Usage:
  python main.py
  → Interactive mode: type any IT task in plain English

  python main.py "reset password for john@company.com to NewPass@99"
  → Single task mode: run one task and exit

Examples of what you can type:
  reset password for john@company.com to Temp@1234
  create a new user alice@company.com named Alice Smith in Engineering
  delete user bob@company.com
  assign a Pro Microsoft 365 license to john@company.com
  check if alice@company.com exists, if not create them and assign a Pro Slack license
"""

import sys
import os
from dotenv import load_dotenv

# Add agent/ directory to path so we can import agent and tasks
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

from tasks import build_task_prompt
from browser_agent import run


BANNER = """
╔══════════════════════════════════════════════════════╗
║          🖥️  IT Admin Agent — browser-use            ║
║  Type a plain-English IT request and watch the       ║
║  agent complete it in a real browser.                ║
╚══════════════════════════════════════════════════════╝

Supported tasks:
  • reset password for <email> to <password>
  • create user <email> named <name> in <department>
  • delete user <email>
  • assign a <plan> <software> license to <email>
  • revoke <software> license from <email>
  • check if <email> exists
  • check if <email> exists, if not create them and assign a <plan> <software> license

Type 'quit' or 'exit' to stop.
Type 'help' to see this menu again.
"""


def main():
    print(BANNER)

    # Single task mode (passed as CLI argument) 
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        _execute(user_input)
        return

    # Interactive mode 
    while True:
        try:
            user_input = input("\n📋 IT Request > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "help":
            print(BANNER)
            continue

        _execute(user_input)


def _execute(user_input: str):
    """Build task prompt and run the agent."""
    print(f"\n Request received: {user_input}")

    # Convert natural language → structured task prompt
    task_prompt = build_task_prompt(user_input)

    print("\n Task prompt built. Launching agent...\n")

    try:
        result = run(task_prompt)
        print(" AGENT RESULT:")
        print(result) 
    except ValueError as e:
        # Missing API key or config error
        print(f"\n Configuration error: {e}")
    except Exception as e:
        print(f"\n Agent error: {e}")
        raise

if __name__ == "__main__":
    main()
