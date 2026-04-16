"""
agent.py
────────
Core browser-use agent setup.

browser-use wraps Playwright with an LLM  to create an agent
that navigates real browsers by seeing screenshots and deciding actions.

The agent.run(task) method:
  1. Takes a screenshot of the current browser state
  2. Sends screenshot + task description to LLM
  3. LLM returns an action: click(x,y), type("..."), navigate(url), done()
  4. browser-use executes the action
  5. Loop back to 1 until llm calls done()
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
    return ChatGroq(model="llama-3.3-70b-versatile",api_key=api_key,temperature=0.0,)

def get_browser() -> Browser:
    config = BrowserConfig(headless=False)
    return Browser(config=config)

async def run_task(task_prompt: str) -> str:
    llm     = get_llm()
    browser = get_browser()
    agent = Agent(task=task_prompt,llm=llm,browser=browser,max_actions_per_step=15,use_vision=False)
    print(task_prompt)
    print()
    result = await agent.run()
    final_output = result.final_result()
    return final_output or "Task completed (no explicit result returned)."


def run(task_prompt: str) -> str:
    import asyncio
    return asyncio.run(run_task(task_prompt))
