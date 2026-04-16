"""
api.py
──────
Wraps the agent in a FastAPI HTTP server.
POST /run-task  →  runs the agent with the given task string
GET  /health    →  health check
"""

from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Ensure agent directory is in path
sys.path.insert(0, os.path.dirname(__file__))

from tasks import build_task_prompt
from agent import run
import uvicorn

app = FastAPI(title="IT Agent API")

class TaskRequest(BaseModel):
    task: str

class TaskResponse(BaseModel):
    result: str
    task_prompt: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run-task", response_model=TaskResponse)
async def run_task(request: TaskRequest):
    task_prompt = build_task_prompt(request.task)
    result = run(task_prompt)
    return TaskResponse(result=result, task_prompt=task_prompt)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
