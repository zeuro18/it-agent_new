"""
agent.py
────────
Core browser-use agent setup.

browser-use wraps Playwright with an LLM (Claude here) to create an agent
that navigates real browsers by seeing screenshots and deciding actions.

Architecture:
  ChatAnthropic (Claude)  ← LLM that reasons about what to do next
        ↓
  browser_use.Agent       ← orchestrates the see → think → act loop
        ↓
  Playwright (Chromium)   ← real browser that executes the actions

The agent.run(task) method:
  1. Takes a screenshot of the current browser state
  2. Sends screenshot + task description to Claude (vision)
  3. Claude returns an action: click(x,y), type("..."), navigate(url), done()
  4. browser-use executes the action
  5. Loop back to 1 until Claude calls done()
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from browser_use import Agent, BrowserConfig, Browser
load_dotenv()

def get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to agent/.env\n")
    return ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",api_key=api_key,temperature=0.0,)

def get_browser() -> Browser:
    is_production = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RENDER")

    config = BrowserConfig(headless=True if is_production else False)
    return Browser(config=config)

async def run_task(task_prompt: str) -> str:
    llm     = get_llm()
    browser = get_browser()
    agent = Agent(task=task_prompt,llm=llm,browser=browser,max_actions_per_step=15,)
    print(task_prompt)
    print()
    result = await agent.run()
    final_output = result.final_result()
    return final_output or "Task completed (no explicit result returned)."


def run(task_prompt: str) -> str:
    return asyncio.run(run_task(task_prompt))
