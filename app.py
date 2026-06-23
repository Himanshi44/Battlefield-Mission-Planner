"""
Battlefield Mission Planner — Main Streamlit App
Architecture: LangGraph multi-agent pipeline with Groq LLM
"""

import streamlit as st
import time
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BATTLENET AI — Mission Command",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Share Tech Mono', monospace !important; }
.stApp { background: #0a0d0f !important; }

.mil-header { background: linear-gradient(90deg,#0d1f0f,#051015,#0d1f0f); border:1px solid #00ff8855;
  border-left:4px solid #00ff88; padding:16px 24px; margin-bottom:20px; border-radius:4px; }
.mil-header h1 { color:#00ff88; font-family:'Rajdhani',sans-serif; font-weight:700;
  font-size:2rem; letter-spacing:6px; margin:0; text-shadow:0 0 20px #00ff8833; }
.mil-header p { color:#8899aa; font-size:0.8rem; margin:4px 0 0; letter-spacing:2px; }

.agent-card { background:#111417; border:1px solid #2a3540; border-left:3px solid #4a5568;
  padding:8px 12px; margin:4px 0; border-radius:2px; }
.agent-card.active { border-left-color:#00ff88; background:#0d1f15; }
.agent-card.done   { border-left-color:#00ccff; opacity:0.8; }
.agent-card.error  { border-left-color:#ff4444; background:#1f0d0d; }
.agent-name  { color:#00ccff; font-weight:700; font-size:0.85rem; letter-spacing:1px; }
.agent-type  { color:#4a5568; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; }
.agent-status{ color:#8899aa; font-size:0.75rem; margin-top:2px; }

.thinking-block { background:#051015; border:1px solid #00ccff44; border-left:3px solid #00ccff;
  padding:12px; margin:8px 0; border-radius:2px; }
.thinking-label { color:#00ccff; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;
  margin-bottom:6px; font-weight:700; }
.result-block   { background:#0d1f15; border:1px solid #00ff8844; border-left:3px solid #00ff88;
  padding:12px; margin:8px 0; border-radius:2px; }
.result-label   { color:#00ff88; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;
  margin-bottom:6px; font-weight:700; }
.warn-block { background:#1a1200; border:1px solid #ffaa0044; border-left:3px solid #ffaa00;
  padding:12px; margin:8px 0; border-radius:2px; }
.warn-label { color:#ffaa00; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;
  margin-bottom:6px; font-weight:700; }
.sim-block  { background:#0a0d0f; border:1px solid #ff444444; border-left:3px solid #ff4444;
  padding:12px; margin:8px 0; border-radius:2px; }
.sim-label  { color:#ff4444; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;
  margin-bottom:6px; font-weight:700; }

.metric-box { background:#111417; border:1px solid #2a3540; padding:10px; border-radius:4px; text-align:center; margin:2px; }
.metric-val { color:#00ff88; font-size:1.6rem; font-weight:700; font-family:'Rajdhani',sans-serif; }
.metric-lbl { color:#4a5568; font-size:0.65rem; letter-spacing:2px; text-transform:uppercase; }

.log-container { max-height:260px; overflow-y:auto; background:#0a0d0f;
  border:1px solid #1a2028; padding:6px; border-radius:2px; }
.log-line         { padding:1px 0; font-size:0.75rem; color:#4a5568; border-bottom:1px solid #1a2028; }
.log-line.success { color:#00ff88; }
.log-line.info    { color:#00ccff; }
.log-line.warn    { color:#ffaa00; }
.log-line.error   { color:#ff4444; }
.log-line.system  { color:#c8d8e8; font-weight:700; }

div.stButton > button { background:#0d1f15 !important; border:1px solid #00ff88 !important;
  color:#00ff88 !important; font-family:'Share Tech Mono',monospace !important;
  letter-spacing:2px !important; text-transform:uppercase !important;
  font-weight:700 !important; border-radius:2px !important; }
div.stButton > button:hover { background:#00ff88 !important; color:#0a0d0f !important; }

section[data-testid="stSidebar"] { background:#0d1117 !important; border-right:1px solid #2a3540; }
.sidebar-hdr { color:#00ccff; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;
  margin:10px 0 5px; padding-bottom:3px; border-bottom:1px solid #2a3540; }

div.stTextArea textarea { background:#111417 !important; border:1px solid #2a3540 !important;
  color:#c8d8e8 !important; font-family:'Share Tech Mono',monospace !important; font-size:0.82rem !important; border-radius:2px !important; }
div.stTextArea textarea:focus { border-color:#00ff88 !important; }

pre { color:#c8d8e8; white-space:pre-wrap; word-break:break-word; margin:0; font-size:0.8rem;
  font-family:'Share Tech Mono',monospace; line-height:1.7; }

.status-badge { display:inline-block; padding:3px 10px; font-size:0.7rem; letter-spacing:2px;
  font-weight:700; text-transform:uppercase; border-radius:2px; }
.s-standby { background:#1a2028; color:#4a5568; border:1px solid #2a3540; }
.s-active  { background:#0d1f15; color:#00ff88; border:1px solid #00ff8866; }
.s-complete{ background:#051015; color:#00ccff; border:1px solid #00ccff66; }
.s-error   { background:#1f0d0d; color:#ff4444; border:1px solid #ff444466; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────
def init_state():
    defaults = {
        "logs": [],
        "agent_states": {k: "standby" for k in [
            "mission_planner","htn_decomposer","recon_agent",
            "fire_support_agent","logistics_agent","mcts_optimizer","execution_simulator"
        ]},
        "results": {},
        "metrics": {"tasks": "—", "iters": "—", "score": "—", "verdict": "—"},
        "pipeline_status": "standby",
        "progress": 0,
        "running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def add_log(msg, level="default"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"ts": ts, "msg": msg, "level": level})

PRESETS = {
    "— Select preset or type custom —": "",
    "🔴 Urban Assault — Seize City Hall": (
        "MISSION: Urban Assault — Alpha Company must seize City Hall in downtown Ravengrad. "
        "Enemy forces have fortified the building with 3 machine gun nests and an IED network. "
        "We have 2 infantry platoons, light armor (4 APCs), and drone support. "
        "Minimize civilian casualties. Objective must be secured within 2 hours."),
    "🟡 Recon — Enemy Supply Depot": (
        "MISSION: Covert Recon — SPECOPS team of 4 must infiltrate and document enemy supply "
        "depot at grid 447-882. Depot defended by 12 guards, CCTV, patrol dogs. "
        "No hostile contact authorized. Exfil by 0400. Intel on fuel stockpiles required."),
    "🔵 Hostage Rescue — Mountain Compound": (
        "MISSION: Hostage Rescue — 3 civilian hostages held in mountain compound (alt. 2200m). "
        "15 enemy combatants, 2 snipers on ridge. Delta team (8 operators) + 2 attack helicopters. "
        "Hostages must NOT be harmed. Weather: low visibility, 40 knot winds."),
    "⚪ Air Superiority — Sector 9": (
        "MISSION: Establish and maintain air superiority over industrial zone Sector 9 for "
        "6-hour window to enable ground advance. Enemy has 2 SAM batteries, MiG-29 squadron. "
        "We have 4 F-16s, 2 EA-18G Growlers, AWACS support."),
    "🟠 Bridge Seizure — River Crossing": (
        "MISSION: Seize and hold Volkov Bridge to enable armored advance across Kazan River. "
        "Bridge guarded by reinforced platoon (40 troops) with AT missiles. We have airborne "
        "company (120 troops), 2 CH-47 Chinooks, and artillery support. Bridge must stay intact."),
}

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mil-header">
  <h1>⬡ BATTLENET AI — MISSION COMMAND</h1>
  <p>MULTI-AGENT TACTICAL PLANNING SYSTEM // LANGGRAPH + GROQ // HTN + MCTS</p>
</div>""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-hdr">⬡ SYSTEM STATUS</div>', unsafe_allow_html=True)
    ps = st.session_state.pipeline_status
    sc = {"standby":"s-standby","active":"s-active","complete":"s-complete","error":"s-error"}.get(ps,"s-standby")
    st.markdown(f'<span class="status-badge {sc}">{ps.upper()}</span>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-hdr">⬡ API CONFIGURATION</div>', unsafe_allow_html=True)
    try:
        from utils.config import GROQ_API_KEY as DEFAULT_KEY
    except Exception:
        DEFAULT_KEY = ""
    api_key = st.text_input("GROQ API KEY", value=DEFAULT_KEY, type="password", placeholder="gsk_...")
    model_choice = st.selectbox("LLM MODEL", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])

    st.markdown('<div class="sidebar-hdr">⬡ MCTS PARAMETERS</div>', unsafe_allow_html=True)
    mcts_iterations = st.slider("Iterations", 10, 100, 40, 5)
    mcts_C = st.slider("Exploration C (UCB1)", 0.5, 2.5, 1.414, 0.1)

    st.markdown('<div class="sidebar-hdr">⬡ AGENT NETWORK</div>', unsafe_allow_html=True)
    AGENTS = [
        ("Mission Planner","Orchestrator","mission_planner",False),
        ("HTN Decomposer","Planner","htn_decomposer",False),
        ("Recon Agent","Specialist","recon_agent",True),
        ("Fire Support Agent","Specialist","fire_support_agent",True),
        ("Logistics Agent","Specialist","logistics_agent",True),
        ("MCTS Optimizer","Decision","mcts_optimizer",False),
        ("Execution Simulator","Simulator","execution_simulator",False),
    ]
    for name, atype, key, indent in AGENTS:
        s = st.session_state.agent_states[key]
        css = {"standby":"","active":" active","done":" done","error":" error"}.get(s,"")
        stxt = {"standby":"Idle","active":"⚡ Running...","done":"✓ Complete","error":"✗ Error"}.get(s,s)
        pad = "padding-left:16px;" if indent else ""
        st.markdown(f"""<div class="agent-card{css}" style="{pad}">
<div class="agent-type">{atype}</div><div class="agent-name">{name}</div>
<div class="agent-status">{stxt}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-hdr">⬡ LANGGRAPH FLOW</div>', unsafe_allow_html=True)
    with st.expander("View Pipeline"):
        st.code("""START
  ↓
mission_planner_node
  ↓
htn_decomposer_node
  ↓ (Send() — parallel)
 ├→ recon_agent_node
 ├→ fire_support_node
 └→ logistics_node
  ↓ (join)
mcts_optimizer_node
  ↓
execution_simulator_node
  ↓
END""", language="text")

# ─── Main Layout ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.55])

with left:
    st.markdown('<p style="color:#00ccff;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;">⬡ MISSION BRIEF</p>', unsafe_allow_html=True)

    preset_sel = st.selectbox("Preset", list(PRESETS.keys()), label_visibility="collapsed")
    default_txt = PRESETS[preset_sel]

    mission_brief = st.text_area("Brief", value=default_txt, height=160,
        placeholder="Describe mission objective, terrain, available units, constraints, ROE...",
        label_visibility="collapsed")

    run_btn = st.button("▶ EXECUTE PLANNING SEQUENCE", use_container_width=True)

    st.markdown('<p style="color:#4a5568;font-size:0.7rem;letter-spacing:1px;margin-top:10px;">PIPELINE PROGRESS</p>', unsafe_allow_html=True)
    prog_bar = st.progress(st.session_state.progress)

    st.markdown('<p style="color:#00ccff;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;margin-top:14px;">⬡ TELEMETRY</p>', unsafe_allow_html=True)
    mc1,mc2,mc3,mc4 = st.columns(4)
    with mc1: st.markdown(f'<div class="metric-box"><div class="metric-val">7</div><div class="metric-lbl">Agents</div></div>', unsafe_allow_html=True)
    with mc2: st.markdown(f'<div class="metric-box"><div class="metric-val">{st.session_state.metrics["tasks"]}</div><div class="metric-lbl">Tasks</div></div>', unsafe_allow_html=True)
    with mc3: st.markdown(f'<div class="metric-box"><div class="metric-val">{st.session_state.metrics["iters"]}</div><div class="metric-lbl">MCTS</div></div>', unsafe_allow_html=True)
    with mc4:
        verd = st.session_state.metrics["verdict"]
        vc = {"GO":"#00ff88","NO-GO":"#ff4444","MODIFY":"#ffaa00","—":"#4a5568"}.get(verd,"#4a5568")
        st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{vc}">{verd}</div><div class="metric-lbl">Verdict</div></div>', unsafe_allow_html=True)

    st.markdown('<p style="color:#00ccff;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;margin-top:14px;">⬡ OPERATION LOG</p>', unsafe_allow_html=True)
    lmap = {"default":"log-line","success":"log-line success","info":"log-line info",
            "warn":"log-line warn","error":"log-line error","system":"log-line system"}
    lhtml = '<div class="log-container">'
    for e in st.session_state.logs[-50:]:
        cls = lmap.get(e["level"],"log-line")
        lhtml += f'<div class="{cls}"><span style="color:#2a3540">{e["ts"]}</span> {e["msg"]}</div>'
    lhtml += '</div>'
    st.markdown(lhtml, unsafe_allow_html=True)

with right:
    st.markdown('<p style="color:#00ccff;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;">⬡ MISSION OUTPUT</p>', unsafe_allow_html=True)

    results = st.session_state.results

    if not results and not run_btn:
        st.markdown("""<div style="text-align:center;padding:60px 20px">
<div style="font-size:3rem;color:#2a3540;margin-bottom:12px">◈</div>
<div style="color:#4a5568;letter-spacing:3px;font-size:0.85rem">AWAITING MISSION BRIEF</div>
<div style="color:#2a3540;font-size:0.75rem;margin-top:8px;line-height:1.9">
Select a preset or enter a custom mission brief.<br>
The pipeline activates 7 specialized AI agents across<br>
HTN planning, MCTS optimization, and execution simulation.
</div></div>""", unsafe_allow_html=True)

    if "planner" in results:
        st.markdown(f"""<div class="thinking-block"><div class="thinking-label">◈ Mission Planner Agent — State Extraction</div>
<pre>{results['planner']}</pre></div>""", unsafe_allow_html=True)

    if "htn" in results:
        st.markdown(f"""<div class="result-block"><div class="result-label">⬡ HTN Decomposition Tree</div>
<pre>{results['htn']}</pre></div>""", unsafe_allow_html=True)

    if "recon" in results or "fire" in results or "logistics" in results:
        st.markdown('<div class="thinking-label" style="color:#00ccff;font-size:0.7rem;letter-spacing:2px;margin:6px 0 2px;">◈ SPECIALIST AGENT REPORTS — PARALLEL EXECUTION</div>', unsafe_allow_html=True)
        s1,s2,s3 = st.columns(3)
        if "recon" in results:
            with s1:
                with st.expander("🔍 Recon"):
                    st.code(results['recon'], language="text")
        if "fire" in results:
            with s2:
                with st.expander("🔥 Fire Support"):
                    st.code(results['fire'], language="text")
        if "logistics" in results:
            with s3:
                with st.expander("📦 Logistics"):
                    st.code(results['logistics'], language="text")

    if "mcts" in results:
        st.markdown(f"""<div class="warn-block"><div class="warn-label">⬡ MCTS Decision Engine — Optimal Action Plan</div>
<pre>{results['mcts']}</pre></div>""", unsafe_allow_html=True)

    if "simulation" in results:
        st.markdown(f"""<div class="sim-block"><div class="sim-label">◈ Execution Simulation — Final Report</div>
<pre>{results['simulation']}</pre></div>""", unsafe_allow_html=True)

    if "summary" in results:
        st.markdown(f"""<div class="result-block"><div class="result-label">⬡ MISSION PLAN FINALIZED</div>
<pre>{results['summary']}</pre></div>""", unsafe_allow_html=True)

# ─── Run Pipeline ─────────────────────────────────────────────────────────────
if run_btn:
    if not mission_brief.strip():
        st.error("⚠ No mission brief provided.")
        st.stop()
    if not api_key or api_key.strip() in ("", "your_groq_api_key_here"):
        st.error("⚠ Groq API key required. Add it in the sidebar or set GROQ_API_KEY in .env")
        st.stop()

    # Reset
    st.session_state.results = {}
    st.session_state.logs = []
    st.session_state.pipeline_status = "active"
    st.session_state.progress = 0
    for k in st.session_state.agent_states:
        st.session_state.agent_states[k] = "standby"
    st.session_state.metrics = {"tasks":"—","iters":"—","score":"—","verdict":"—"}

    add_log("Mission brief received", "system")
    add_log(f"Model: {model_choice} | MCTS: {mcts_iterations} iter", "info")
    add_log("Initializing LangGraph pipeline...", "info")

    from agents.pipeline import run_pipeline

    t0 = time.time()

    def update_progress(pct, msg, level="default"):
        prog_bar.progress(pct)
        st.session_state.progress = pct
        add_log(msg, level)

    def set_agent(key, status):
        st.session_state.agent_states[key] = status

    results = run_pipeline(
        mission_brief=mission_brief,
        api_key=api_key,
        model=model_choice,
        mcts_iterations=mcts_iterations,
        mcts_C=mcts_C,
        progress_cb=update_progress,
        agent_cb=set_agent,
    )

    st.session_state.results = results["outputs"]
    st.session_state.metrics = results["metrics"]
    st.session_state.pipeline_status = "complete"

    elapsed = round(time.time() - t0, 1)
    add_log(f"Pipeline complete in {elapsed}s", "system")
    add_log(f"Verdict: {results['metrics']['verdict']}", "success")

    st.rerun()
