import os
import json
import time
from datetime import datetime
import streamlit as st
from openai import OpenAI

NVIDIA_API_KEY = "[INSERT_NVIDIA_API_KEY_HERE]"
MEMORY_FILE = "scout_memory.json"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

def robust_api_call(prompt, temperature=0.1, max_tokens=700, max_retries=3):
    """Wraps the API call in a retry loop to survive server drops."""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = completion.choices[0].message.content
            
            if content and content.strip():
                return content
            else:
                time.sleep(1.5)
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2)
            
    return None

def read_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def write_memory(player_id, log_entry):
    if "API returned empty response" in log_entry or "Error" in log_entry:
        return 
        
    memory = read_memory()
    if player_id not in memory:
        memory[player_id] = []

    memory[player_id].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": log_entry
    })

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

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
Previous Player History: {history}
Live Observation: {observation}

Respond EXACTLY in this format:
THOUGHT: [reasoning about the player's condition]
ACTION: [ALERT_COACH or LOG_ONLY]
SUMMARY: [one-sentence summary]
"""

    content = robust_api_call(prompt, temperature=0.1, max_tokens=700)
    
    if not content:
        return "THOUGHT: API server overloaded after 3 retries.\nACTION: LOG_ONLY\nSUMMARY: Agent failed to connect to Nemotron."
    return content

def generate_team_report():
    memory = read_memory()

    if not memory:
        return "No scout memory available yet. Run the agent on a few player observations first."

    prompt = f"""
You are HoopsInsight, an autonomous basketball scouting agent.
Generate a comprehensive team report using the full persistent scout memory below.

Scout Memory:
{json.dumps(memory, indent=2)}

Report format:
TEAM REPORT:
- Overall Team Status:
- Players Needing Attention:
- Positive Trends:
- Recommended Coaching Actions:
"""
    content = robust_api_call(prompt, temperature=0.2, max_tokens=1500)
    
    if not content:
        return "Report generation failed. API server overloaded after 3 retries."
    return content

def predict_season_stats(player_name, games_played, games_remaining, mpg, ppg, rpg, apg, spg, bpg):
    memory = read_memory()
    player_history = memory.get(player_name, "No previous scouting history available.")

    prompt = f"""
You are HoopsInsight, an autonomous basketball stats projection agent.

Player: {player_name}

Current sample:
Games played: {games_played}
Games remaining: {games_remaining}

Current averages:
MPG: {mpg}
PPG: {ppg}
RPG: {rpg}
APG: {apg}
SPG: {spg}
BPG: {bpg}

Persistent scouting history for this player:
{player_history}

Your task:
Predict REST-OF-SEASON AVERAGES, not totals.

Important:
- Do NOT simply copy the current averages.
- Adjust the projection using the scouting history.
- If the scouting history mentions fatigue, injury risk, defensive slowdown, or poor conditioning, slightly lower MPG/PPG and explain why.
- If the scouting history mentions recovery, strong rhythm, efficient shooting, or improved movement, slightly raise or stabilize the projection.
- Keep projections realistic. Small changes are better than huge changes.

Return ONLY valid JSON in this exact format:

{{
    "projected_mpg": number,
    "projected_ppg": number,
    "projected_rpg": number,
    "projected_apg": number,
    "projected_spg": number,
    "projected_bpg": number,
    "justification": "short explanation that references the scouting history",
    "coaching_insight": "short coaching insight"
}}

Do not return markdown.
Do not return code blocks.
Do not leave fields empty.
"""

    content = robust_api_call(prompt, temperature=0.35, max_tokens=800)

    if not content:
        raise ValueError("API server overloaded after 3 retries.")

    try:
        return json.loads(content)
    except Exception:
        return {
            "projected_mpg": round(mpg * 0.97, 1),
            "projected_ppg": round(ppg * 0.97, 1),
            "projected_rpg": round(rpg, 1),
            "projected_apg": round(apg, 1),
            "projected_spg": round(spg, 1),
            "projected_bpg": round(bpg, 1),
            "justification": "Model formatting failed. Applied conservative adjustment instead of copying baseline averages.",
            "coaching_insight": "Use scouting memory to monitor whether performance trend improves or worsens."
        }


def parse_agent_response(response):
    action = "LOG_ONLY"
    summary = "No summary generated."

    if not response:
        return action, "Error: No response generated by the API."

    for line in response.splitlines():
        if line.startswith("ACTION:"):
            action = line.replace("ACTION:", "").strip()
        elif line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()

    return action, summary

st.set_page_config(page_title="HoopsInsight", page_icon="🏀", layout="wide")

if "player_count" not in st.session_state:
    st.session_state.player_count = 1

st.title("🏀 HoopsInsight")
st.caption("Autonomous Basketball Scouting Agent powered by NVIDIA Nemotron")

left, right = st.columns([1, 1])

with left:
    st.subheader("Live Scout Observations")

    col_add, col_rem = st.columns(2)
    with col_add:
        if st.button("➕ Add Player", use_container_width=True):
            st.session_state.player_count += 1
            st.rerun()
    with col_rem:
        if st.button("➖ Remove Player", use_container_width=True):
            if st.session_state.player_count > 1:
                st.session_state.player_count -= 1
                st.rerun()

    for i in range(st.session_state.player_count):
        st.markdown(f"**Observation {i+1}**")
        st.text_input("Player ID", key=f"pid_{i}")
        st.text_area("Courtside Observation", key=f"obs_{i}", height=100)

    run_button = st.button("Run Autonomous Agent on All", use_container_width=True, type="primary")

    st.divider()
    st.subheader("Quick Demo Inputs")

    if st.button("Load Demo Scenario", use_container_width=True):
        st.session_state.player_count = 3
        st.session_state["pid_0"] = "Player_7"
        st.session_state["obs_0"] = "Missed 6 straight shots, is slow getting back on defense, looks visibly exhausted."
        st.session_state["pid_1"] = "Stephen_Curry"
        st.session_state["obs_1"] = "Quick release still looks elite. Hit 4 consecutive threes in transition offense."
        st.session_state["pid_2"] = "Nikola_Jokic"
        st.session_state["obs_2"] = "Distributing the ball perfectly from the high post, but struggling to guard the pick and roll."
        st.rerun()

with right:
    st.subheader("Agent Output")

    if run_button:
        with st.spinner("Nemotron agent reasoning for all players... (This may take a moment)"):
            for i in range(st.session_state.player_count):
                pid = st.session_state.get(f"pid_{i}")
                obs = st.session_state.get(f"obs_{i}")

                if pid and obs:
                    with st.expander(f"Results: {pid}", expanded=True):
                        try:
                            response = run_agent(pid, obs)
                            action, summary = parse_agent_response(response)

                            st.markdown("### ReAct Reasoning")
                            st.code(response, language="text")

                            write_memory(pid, summary)

                            st.markdown("### Autonomous Action")
                            if "ALERT_COACH" in action:
                                st.error("ACTION: ALERT_COACH")
                                st.info("Coach alert recommended.")
                            else:
                                st.success(f"ACTION: {action}")
                                st.info("Agent chose not to alert the coach.")

                            st.markdown("### Memory Updated")
                            st.success(summary)

                        except Exception as e:
                            st.error(f"Agent failed for {pid}: {e}")

st.divider()
st.subheader("Persistent Scout Memory")

memory = read_memory()
if memory:
    st.json(memory)
else:
    st.info("No memory yet. Run the agent once to create persistent player history.")

st.divider()
st.subheader("Team Report")

if st.button("Generate Team Report", use_container_width=True):
    with st.spinner("Nemotron generating comprehensive team report..."):
        try:
            report = generate_team_report()
            st.code(report, language="text")
        except Exception as e:
            st.error(f"Team report failed: {e}")

st.divider()
st.subheader("Season Average Prediction")

with st.form("stats_prediction_form"):
    stat_player_name = st.text_input("Player Name / ID", value="Player_7")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        games_played = st.number_input("Games Played", min_value=1, value=10)
        games_remaining = st.number_input("Games Remaining", min_value=1, value=15)
    with col_b:
        mpg = st.number_input("Minutes Per Game", min_value=0.0, value=30.0)
        ppg = st.number_input("Points Per Game", min_value=0.0, value=18.5)
        rpg = st.number_input("Rebounds Per Game", min_value=0.0, value=6.2)
    with col_c:
        apg = st.number_input("Assists Per Game", min_value=0.0, value=4.1)
        spg = st.number_input("Steals Per Game", min_value=0.0, value=1.3)
        bpg = st.number_input("Blocks Per Game", min_value=0.0, value=0.7)

    predict_button = st.form_submit_button("Predict Rest-of-Season Averages", use_container_width=True)

if predict_button:
    with st.spinner("Nemotron projecting rest-of-season averages..."):
        try:
            prediction = predict_season_stats(
                stat_player_name,
                games_played,
                games_remaining,
                mpg,
                ppg,
                rpg,
                apg,
                spg,
                bpg
            )

            st.markdown("### Memory Used For Prediction")
            st.json(
                read_memory().get(
                    stat_player_name,
                    "No previous scouting history available."
                )
            )

            st.markdown("### Rest-of-Season Average Prediction")

            st.metric("Projected MPG", prediction["projected_mpg"])
            st.metric("Projected PPG", prediction["projected_ppg"])
            st.metric("Projected RPG", prediction["projected_rpg"])
            st.metric("Projected APG", prediction["projected_apg"])
            st.metric("Projected SPG", prediction["projected_spg"])
            st.metric("Projected BPG", prediction["projected_bpg"])

            st.markdown("### Justification")
            st.info(prediction["justification"])

            st.markdown("### Coaching Insight")
            st.success(prediction["coaching_insight"])

        except Exception as e:
            st.error(f"Stats prediction failed: {e}")
            
if st.sidebar.button("Clear Persistent Memory"):
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
        st.rerun()