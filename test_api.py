import requests
import json

# REPLACE THIS WITH YOUR AGENT'S RAILWAY DOMAIN:
AGENT_URL = "https://agent-production-2c26.up.railway.app"

def test_agent(task_text):
    print(f"Sending task to agent: '{task_text}'...\n")
    
    url = f"{AGENT_URL}/run-task"
    
    # Send the POST request to the FastAPI endpoint
    response = requests.post(
        url,
        json={"task": task_text},
        timeout=180 # The agent takes time to open chromium and navigate, so we wait
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS!")
        print("-" * 40)
        print("Task Prompt Built:")
        print(data.get("task_prompt", ""))
        print("\nAgent Execution Result:")
        print(data.get("result", ""))
    else:
        print(f"❌ ERROR: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    task = "create a new user alice@decawork.com named Alice in Engineering"
    test_agent(task)
