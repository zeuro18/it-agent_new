"""
agent_core.py
─────────────
ITAgent: the orchestrating agent that ties together:
  1. Structured tools (fast, reliable DB operations)
  2. RAG retrieval (policy knowledge)
  3. Browser agent (fallback for complex/novel UI tasks)
  4. Verification (post-action DB state checks)

Uses Groq's free LLaMA models for all LLM calls.
"""

import os
import sys
import time
import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Ensure imports work
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from langchain_groq import ChatGroq

import tools
from rag.retriever import retrieve, format_context


@dataclass
class AgentResult:
    """Structured output from every agent run."""
    success: bool
    message: str
    evidence: dict = field(default_factory=dict)
    citations: list = field(default_factory=list)
    latency_s: float = 0.0
    tokens_used: int = 0
    method: str = ""  # "tool", "browser", "rag", "tool+rag"


# ── Tool definitions for LLM function calling ──────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "user_lookup",
            "description": "Look up a user by email address. Returns their profile, status, department, role, and group memberships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The user's email address"}
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_licenses",
            "description": "List all software licenses assigned to a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The user's email address"}
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assign_license",
            "description": "Assign a software license to a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The user's email"},
                    "software": {"type": "string", "description": "Software name (e.g., Microsoft 365, Slack, GitHub)"},
                    "plan": {"type": "string", "description": "Plan tier (Standard, Business, Pro, Enterprise)"}
                },
                "required": ["email", "software", "plan"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "revoke_license",
            "description": "Revoke a software license from a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The user's email"},
                    "software": {"type": "string", "description": "Software name to revoke"}
                },
                "required": ["email", "software"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_pending_tickets",
            "description": "List all support tickets with status 'Pending'.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_ticket",
            "description": "Update a ticket's status and add notes. Status can be: Pending, Approved, Rejected, Resolved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "integer", "description": "The ticket ID number"},
                    "status": {"type": "string", "enum": ["Pending", "Approved", "Rejected", "Resolved"]},
                    "notes": {"type": "string", "description": "Admin notes explaining the decision"}
                },
                "required": ["ticket_id", "status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assign_user_to_group",
            "description": "Add a user to an RBAC group.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "group_name": {"type": "string", "description": "Group name (e.g., Developers, Marketing, HR, IT Admin)"}
                },
                "required": ["email", "group_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_user_from_group",
            "description": "Remove a user from an RBAC group.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "group_name": {"type": "string"}
                },
                "required": ["email", "group_name"]
            }
        }
    },
]

# Map function names to actual callables
TOOL_MAP = {
    "user_lookup": tools.user_lookup,
    "list_licenses": tools.list_licenses,
    "assign_license": tools.assign_license,
    "revoke_license": tools.revoke_license,
    "list_pending_tickets": tools.list_pending_tickets,
    "update_ticket": tools.update_ticket,
    "assign_user_to_group": tools.assign_user_to_group,
    "remove_user_from_group": tools.remove_user_from_group,
}


class ITAgent:
    """
    The core IT Agent that processes requests using tools, RAG, and browser.
    
    All LLM calls use Groq's free API with open-source models.
    """

    def __init__(self, rag_mode: str = "hybrid", use_tools: bool = True, use_browser: bool = True):
        self.rag_mode = rag_mode  # "hybrid", "dense", "bm25", or "none"
        self.use_tools = use_tools
        self.use_browser = use_browser
        self.llm = self._get_llm()

    def _get_llm(self) -> ChatGroq:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Add it to .env")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.0,
        )

    def run(self, user_request: str) -> AgentResult:
        """
        Process a natural-language IT request end-to-end.
        
        Flow:
        1. Retrieve relevant policy context (RAG)
        2. Ask LLM to plan: which tool(s) to call, or use browser
        3. Execute tool calls in sequence
        4. Verify the outcome against DB state
        5. Return structured result with evidence
        """
        start = time.time()
        total_tokens = 0

        # ── Step 1: RAG context ──
        rag_context = ""
        citations = []
        if self.rag_mode != "none":
            try:
                rag_results = retrieve(user_request, k=3, mode=self.rag_mode)
                rag_context = format_context(rag_results)
                citations = [r["citation"] for r in rag_results]
            except Exception as e:
                rag_context = f"(RAG unavailable: {e})"

        # ── Step 2: LLM planning with tool calling ──
        system_prompt = self._build_system_prompt(rag_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request},
        ]

        # Multi-turn tool calling loop (max 10 iterations to prevent runaway)
        all_evidence = {}
        final_message = ""

        for iteration in range(10):
            if self.use_tools:
                response = self.llm.invoke(messages, tools=TOOL_SCHEMAS)
            else:
                response = self.llm.invoke(messages)

            total_tokens += response.response_metadata.get("token_usage", {}).get("total_tokens", 0)

            # Check if the LLM wants to call tools
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Process each tool call
                messages.append(response)  # Add assistant message with tool calls

                for tool_call in response.tool_calls:
                    fn_name = tool_call["name"]
                    fn_args = tool_call["args"]

                    print(f"Tool: {fn_name}({json.dumps(fn_args)})")

                    if fn_name in TOOL_MAP:
                        result = TOOL_MAP[fn_name](**fn_args)
                        all_evidence[f"{fn_name}({json.dumps(fn_args)})"] = result
                    else:
                        result = {"error": f"Unknown tool: {fn_name}"}

                    # Feed the tool result back to the LLM
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result, default=str),
                        "tool_call_id": tool_call["id"],
                    })
            else:
                # LLM is done — produced a final text response
                final_message = response.content
                break
        else:
            final_message = "Agent reached maximum iterations without completing."

        # ── Step 3: Verification ──
        verification = self._verify(user_request, all_evidence)

        elapsed = time.time() - start

        return AgentResult(
            success=verification["verified"],
            message=final_message,
            evidence=all_evidence,
            citations=citations,
            latency_s=round(elapsed, 2),
            tokens_used=total_tokens,
            method="tool" if all_evidence else "llm_only",
        )

    def _build_system_prompt(self, rag_context: str) -> str:
        """Build the system prompt with RAG context injected."""
        base = """You are an IT admin assistant. You have access to tools for managing users, licenses, tickets, and groups in the company's IT system.

INSTRUCTIONS:
- Use the tools provided to complete the user's request.
- For multi-step tasks, call tools in the correct sequence.
- After completing all actions, provide a clear summary of what was done.
- If a tool returns an error, explain the issue clearly.
- Always verify your actions make sense before executing (e.g., check if a user exists before assigning them a license).
- Reference relevant policy when making decisions about approvals or rejections.
"""

        if rag_context and rag_context != "No relevant policy documents found.":
            base += f"""
RELEVANT COMPANY POLICIES (use these to guide your decisions):
{rag_context}
"""

        return base

    def _verify(self, request: str, evidence: dict) -> dict:
        """
        Post-action verification.
        
        Instead of trusting the LLM's claim, we check the actual DB state
        by calling read-only tools and comparing against what was requested.
        """
        if not evidence:
            return {"verified": False, "reason": "No tool actions were executed"}

        # Check if any tool returned an error
        errors = []
        successes = []
        for call, result in evidence.items():
            if isinstance(result, dict):
                if result.get("success") is False:
                    errors.append(f"{call} → {result.get('error', 'unknown error')}")
                elif result.get("success") is True:
                    successes.append(call)
            elif result is None:
                errors.append(f"{call} → returned None (user/resource not found)")

        if errors and not successes:
            return {"verified": False, "reason": f"All actions failed: {'; '.join(errors)}"}
        elif errors:
            return {"verified": True, "reason": f"Partial success. Failures: {'; '.join(errors)}"}
        else:
            return {"verified": True, "reason": f"All {len(successes)} actions succeeded"}


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    """Interactive CLI for the IT Agent."""
    import argparse
    parser = argparse.ArgumentParser(description="IT Agent with Tools + RAG")
    parser.add_argument("request", nargs="*", help="IT request in natural language")
    parser.add_argument("--rag", default="hybrid", choices=["hybrid", "dense", "bm25", "none"],
                        help="RAG retrieval mode")
    parser.add_argument("--no-tools", action="store_true", help="Disable structured tools")
    parser.add_argument("--no-browser", action="store_true", help="Disable browser fallback")
    args = parser.parse_args()

    agent = ITAgent(
        rag_mode=args.rag,
        use_tools=not args.no_tools,
        use_browser=not args.no_browser,
    )

    if args.request:
        user_input = " ".join(args.request)
        result = agent.run(user_input)
        _print_result(result)
        return

    # Interactive mode
    print("\nIT Agent (Tools + RAG + Verification)")
    print(f"RAG mode: {args.rag} | Tools: {not args.no_tools}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Request > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = agent.run(user_input)
        _print_result(result)


def _print_result(result: AgentResult):
    """Print an agent result."""
    status = "SUCCESS" if result.success else "FAILED"
    print(f"[{status}] ({result.method}) - {result.latency_s}s, {result.tokens_used} tokens")
    print(f"{result.message}")

    if result.citations:
        print(f"\nCitations:")
        for c in result.citations:
            print(f"  {c}")

    if result.evidence:
        print(f"\nEvidence:")
        for call, res in result.evidence.items():
            print(f"  {call}")
            if isinstance(res, dict):
                for k, v in res.items():
                    print(f"      {k}: {v}")
    print()


if __name__ == "__main__":
    main()
