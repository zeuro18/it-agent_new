"""
api.py
──────
Wraps the agent in a FastAPI HTTP server.
POST /run-task  →  runs the agent with the given task string
GET  /health    →  health check
"""

from fastapi import FastAPI
from pydantic import BaseModel
from tasks import build_task_prompt
from agent import run_task
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
async def run_task_endpoint(request: TaskRequest):
    task_prompt = build_task_prompt(request.task)
    # Call run_task directly with await — no asyncio.run() needed
    # FastAPI already manages the event loop
    result = await run_task(task_prompt)
    return TaskResponse(result=result, task_prompt=task_prompt)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)