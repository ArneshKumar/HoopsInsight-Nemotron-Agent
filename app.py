import os
import json
import requests
from datetime import datetime
import streamlit as st
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
NVIDIA_API_KEY = ""
DISCORD_WEBHOOK_URL = "" 
MEMORY_FILE = "scout_memory.json"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# ----------------------------
# MEMORY TOOLS
# ----------------------------
def read_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def write_memory(player_id, log_entry):
    memory = read_memory()

    if player_id not in memory:
        memory[player_id] = []

    memory[player_id].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": log_entry
    })

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

def send_discord_alert(message):
    if not DISCORD_WEBHOOK_URL:
        return "Simulated alert: Discord webhook not configured."

    try:
        requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": f"🚨 HoopsInsight Alert: {message}"},
            timeout=5
        )
        return "Discord alert sent successfully."
    except Exception as e:
        return f"Discord alert failed: {e}"

# ----------------------------
# AGENT
# ----------------------------
def run_agent(player_id, observation):
    memory = read_memory()
    history = memory.get(player_id, "No previous history.")

    prompt = f"""
You are HoopsInsight, an autonomous basketball scouting agent.

Your job:
1. Analyze the live basketball observation.
2. Compare it against the player's memory/history.
3. Decide whether the coach should be alerted.
4. Return a structured ReAct-style response.

Player ID: {player_id}

Previous Player History:
{history}

Live Observation:
{observation}

Respond EXACTLY in this format:

THOUGHT: [reasoning about the player's condition]
ACTION: [ALERT_COACH or LOG_ONLY]
SUMMARY: [one-sentence summary]
"""

    completion = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=700
    )

    return completion.choices[0].message.content

def parse_agent_response(response):
    action = "LOG_ONLY"
    summary = "No summary generated."

    for line in response.splitlines():
        if line.startswith("ACTION:"):
            action = line.replace("ACTION:", "").strip()
        elif line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()

    return action, summary

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(
    page_title="HoopsInsight",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 HoopsInsight")
st.caption("Autonomous Basketball Scouting Agent powered by NVIDIA Nemotron")

left, right = st.columns([1, 1])

with left:
    st.subheader("Live Scout Observation")

    player_id = st.text_input("Player ID", value="Player_7")

    observation = st.text_area(
        "Courtside Observation",
        value="Missed 6 straight shots and looks exhausted on defense.",
        height=180
    )

    run_button = st.button("Run Autonomous Agent", use_container_width=True)

    st.divider()

    st.subheader("Quick Demo Inputs")

    if st.button("Fatigue Alert Scenario"):
        observation = "Player 7 missed 6 straight shots, is slow getting back on defense, and looks visibly exhausted."
        st.rerun()

    if st.button("Recovery Scenario"):
        observation = "Player 7 recovered after a timeout, made 3 straight shots, and is moving better defensively."
        st.rerun()

with right:
    st.subheader("Agent Output")

    if run_button:
        with st.spinner("Nemotron agent reasoning..."):
            try:
                response = run_agent(player_id, observation)
                action, summary = parse_agent_response(response)

                st.markdown("### ReAct Reasoning")
                st.code(response, language="text")

                write_memory(player_id, summary)

                st.markdown("### Autonomous Action")

                if action == "ALERT_COACH":
                    st.error("ACTION: ALERT_COACH")
                    alert_status = send_discord_alert(f"{player_id}: {summary}")
                    st.info(alert_status)
                else:
                    st.success("ACTION: LOG_ONLY")
                    st.info("Agent chose not to alert the coach.")

                st.markdown("### Memory Updated")
                st.success(summary)

            except Exception as e:
                st.error(f"Agent failed: {e}")

st.divider()

st.subheader("Persistent Scout Memory")

memory = read_memory()

if memory:
    st.json(memory)
else:
    st.info("No memory yet. Run the agent once to create persistent player history.")