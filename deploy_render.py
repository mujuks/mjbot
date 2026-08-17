import os
import requests

API_KEY = os.environ.get("RENDER_API_KEY", "")
if not API_KEY:
    print("Set RENDER_API_KEY environment variable first")
    exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

payload = {
    "name": "mjbot",
    "ownerId": "tea-da1akv3l550s73f7trbg",
    "type": "web_service",
    "runtime": "python",
    "repo": "https://github.com/mujuks/mjbot.git",
    "branch": "master",
    "autoDeploy": "yes",
    "serviceDetails": {
        "runtime": "python",
        "envSpecificDetails": {
            "buildCommand": "pip install -r requirements.txt",
            "startCommand": "python bot.py",
        },
        "plan": "free",
        "region": "oregon",
        "numInstances": 1,
    },
    "envVars": [
        {"key": "TELEGRAM_BOT_TOKEN", "value": TELEGRAM_BOT_TOKEN},
        {"key": "TELEGRAM_CHAT_ID", "value": TELEGRAM_CHAT_ID},
        {"key": "WEBHOOK_HOST", "value": "0.0.0.0"},
        {"key": "PYTHON_VERSION", "value": "3.12.0"},
    ],
}

print("Creating background worker on Render...")
r = requests.post(
    "https://api.render.com/v1/services",
    headers=HEADERS,
    json=payload,
    timeout=30,
)
print(f"Status: {r.status_code}")

if r.status_code in (200, 201):
    data = r.json()
    svc = data.get("service", data)
    deploy_id = data.get("deployId", "")
    svc_id = svc.get("id", "")
    print(f"\nService created successfully!")
    print(f"  ID: {svc_id}")
    print(f"  Name: {svc.get('name', '')}")
    print(f"  Deploy ID: {deploy_id}")
    print(f"  Dashboard: https://dashboard.render.com/web/{svc_id}")
else:
    print(f"Error: {r.text[:500]}")
