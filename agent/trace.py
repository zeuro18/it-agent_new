"""
trace.py
Structured Execution Trace (Phase 3).

Captures every step of the agent's execution lifecycle (retrieval, tool calls,
policy decisions, verifications, self-repair attempts) for observability,
auditability, and experiment analysis.
"""

from __future__ import annotations
import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class TraceEvent:
    stage: str          # "rag", "policy", "tool", "verify", "repair", "browser"
    timestamp: float    # time.time()
    data: dict          # stage-specific payload
    latency_s: float = 0.0


@dataclass
class AgentTrace:
    task_id: str = ""
    user_request: str = ""
    config: str = ""
    events: list[TraceEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    total_tokens: int = 0
    final_success: bool = False
    repaired: bool = False

    def add(self, stage: str, data: dict, latency_s: float = 0.0):
        """Record an execution event in the trace."""
        self.events.append(
            TraceEvent(
                stage=stage,
                timestamp=time.time(),
                data=data,
                latency_s=round(latency_s, 3),
            )
        )

    def finish(self, success: bool, tokens: int = 0, repaired: bool = False):
        """Mark trace completion."""
        self.end_time = time.time()
        self.final_success = success
        self.total_tokens = tokens
        self.repaired = repaired

    def to_dict(self) -> dict:
        """Convert trace to JSON-serializable dictionary."""
        return {
            "task_id": self.task_id,
            "user_request": self.user_request,
            "config": self.config,
            "duration_s": round(self.end_time - self.start_time, 3) if self.end_time else 0.0,
            "total_tokens": self.total_tokens,
            "final_success": self.final_success,
            "repaired": self.repaired,
            "events": [
                {
                    "stage": e.stage,
                    "timestamp": e.timestamp,
                    "latency_s": e.latency_s,
                    "data": e.data,
                }
                for e in self.events
            ],
        }

    def save(self, directory: str = "eval/results/traces") -> str:
        """Save the trace as a JSON file."""
        os.makedirs(directory, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.task_id}_" if self.task_id else ""
        filename = f"{prefix}{ts}.json"
        path = os.path.join(directory, filename)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path
