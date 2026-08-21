"""
browser_agent.py
Browser-use agent: drives a real Playwright browser against the admin panel.

browser-use wraps Playwright with an LLM. The agent receives the page DOM as
text (use_vision=False), decides the next action (click, type, navigate) via
function calling, executes it, and loops until it calls done().
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from browser_use import Agent, BrowserConfig, Browser

load_dotenv()

# The core agent loop talks to Groq directly with no framework; ChatGroq is
# used only here because browser-use requires a LangChain chat model.


def get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to agent/.env\n")
    return ChatGroq(model=os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"), api_key=api_key, temperature=0.0)


def get_browser(headless: bool = None) -> Browser:
    if headless is None:
        headless = os.getenv("BROWSER_HEADLESS", "0").lower() in ("1", "true")
    config = BrowserConfig(headless=headless)
    return Browser(config=config)


async def run_task(task_prompt: str, headless: bool = None) -> str:
    llm = get_llm()
    browser = get_browser(headless)
    agent = Agent(task=task_prompt, llm=llm, browser=browser, max_actions_per_step=15, use_vision=False)
    result = await agent.run()
    final_output = result.final_result()
    return final_output or "Task completed (no explicit result returned)."


def run(task_prompt: str, headless: bool = None) -> str:
    return asyncio.run(run_task(task_prompt, headless))
