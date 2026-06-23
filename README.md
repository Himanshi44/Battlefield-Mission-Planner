# ⬡ BATTLENET AI — Battlefield Mission Planner

A multi-agent AI system for tactical military mission planning using **LangGraph**, **Groq LLM**, and **Streamlit**.

```
┌──────────────────────────────────────────────────────────────────┐
│  User Mission Brief                                               │
│        ↓                                                          │
│  [1] Mission Planner Agent  (Orchestrator — LangGraph ROOT)      │
│        ↓                                                          │
│  [2] HTN Task Decomposer    (Hierarchical Task Network)          │
│        ↓  ←── LangGraph Send() parallel fan-out                  │
│   ┌────┴────┬────────────┬──────────────┐                        │
│  [3a] Recon [3b] Fire Sup [3c] Logistics  (Specialist Agents)    │
│   └────┬────┴────────────┴──────────────┘                        │
│        ↓  ←── LangGraph join                                      │
│  [4] MCTS Decision Engine   (Monte Carlo Tree Search)            │
│        ↓                                                          │
│  [5] Execution Simulator    (Step-by-step plan validation)       │
│        ↓                                                          │
│  GO / NO-GO / MODIFY Verdict                                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Get a Groq API Key (free)
Visit [console.groq.com](https://console.groq.com), sign up, and create an API key.

### 2. Install & Run

**Linux/Mac:**
```bash
bash run.sh
```

**Windows:**
```
run.bat
```

**Manual:**
```bash
pip install -r requirements.txt
cp .env.example .env          # Edit .env and add your GROQ_API_KEY
streamlit run app.py
```

### 3. Open Browser
Navigate to `http://localhost:8501`

---

## Architecture

### Agent Network (LangGraph StateGraph)

| Node | Agent | Role | LangGraph API |
|------|-------|------|--------------|
| 1 | Mission Planner Agent | Parses mission brief, extracts objectives/terrain/units | Root node — `START → mission_planner` |
| 2 | HTN Decomposer | Hierarchical Task Network — decomposes goals to primitive tasks | Sequential edge |
| 3a | Recon Agent | Intel, terrain analysis, enemy disposition | Parallel via `Send()` |
| 3b | Fire Support Agent | Artillery, air support, target list | Parallel via `Send()` |
| 3c | Logistics Agent | Supply, ammo, MEDEVAC planning | Parallel via `Send()` |
| 4 | MCTS Optimizer | Monte Carlo Tree Search — optimal action sequence | Join point → sequential |
| 5 | Execution Simulator | Step-by-step simulation, GO/NO-GO verdict | Final node → `END` |

### LangGraph Flow (Code)
```python
from langgraph.graph import StateGraph, START, END, Send

graph = StateGraph(AgentState)

# Register nodes
graph.add_node("mission_planner",     mission_planner_fn)
graph.add_node("htn_decomposer",      htn_decomposer_fn)
graph.add_node("recon_agent",         recon_fn)
graph.add_node("fire_support_agent",  fire_fn)
graph.add_node("logistics_agent",     logistics_fn)
graph.add_node("mcts_optimizer",      mcts_fn)
graph.add_node("execution_simulator", simulator_fn)

# Edges
graph.add_edge(START,               "mission_planner")
graph.add_edge("mission_planner",   "htn_decomposer")

# Parallel fan-out via Send() API
def route_to_specialists(state):
    return [
        Send("recon_agent",        state),
        Send("fire_support_agent", state),
        Send("logistics_agent",    state),
    ]
graph.add_conditional_edges("htn_decomposer", route_to_specialists,
    ["recon_agent", "fire_support_agent", "logistics_agent"])

# Join point
graph.add_edge("recon_agent",        "mcts_optimizer")
graph.add_edge("fire_support_agent", "mcts_optimizer")
graph.add_edge("logistics_agent",    "mcts_optimizer")
graph.add_edge("mcts_optimizer",     "execution_simulator")
graph.add_edge("execution_simulator", END)

app = graph.compile()
```

### HTN Planning
Hierarchical Task Network (HTN) decomposes complex goals:
```
MISSION GOAL
  └── Compound Task 1 (e.g. "Secure Perimeter") → RECON agent
        ├── Primitive 1.1: Deploy observation posts
        │     Precond: recon assets available
        │     Effect:  enemy positions known
        └── Primitive 1.2: Establish blocking positions
              Precond: OP-1 and OP-2 established
              Effect:  perimeter secured
  └── Compound Task 2 (e.g. "Suppress Enemy Fire") → FIRE_SUPPORT
        └── ...
```

### MCTS Decision Engine
Monte Carlo Tree Search with UCB1:
```
UCB1(node) = Q/N + C × √(ln(N_parent) / N)
  Q = accumulated reward
  N = visit count
  C = exploration constant (default √2 ≈ 1.414)
```
- **Selection**: Traverse tree choosing max UCB1 at each level
- **Expansion**: LLM generates candidate next actions
- **Rollout**: LLM simulates full action sequence to estimate value
- **Backpropagation**: Update Q and N for all ancestors

---

## Project Structure
```
battlefield_mission_planner/
├── app.py                      # Streamlit main app
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── run.sh                      # Linux/Mac launcher
├── run.bat                     # Windows launcher
├── agents/
│   ├── __init__.py
│   ├── pipeline.py             # LangGraph StateGraph orchestrator
│   ├── mission_planner.py      # Node 1: Orchestrator agent
│   ├── htn_decomposer.py       # Node 2: HTN planning agent
│   ├── specialist_agents.py    # Node 3: Recon + Fire + Logistics
│   ├── mcts_optimizer.py       # Node 4: MCTS decision engine
│   └── execution_simulator.py  # Node 5: Simulation + verdict
└── utils/
    ├── __init__.py
    ├── config.py               # Configuration loader
    ├── groq_client.py          # Groq API client wrapper
    └── state.py                # AgentState + MCTSNode dataclasses
```

---

## Configuration

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
DEFAULT_MODEL=llama-3.3-70b-versatile
MCTS_ITERATIONS=40
MCTS_EXPLORATION_C=1.414
```

Or configure everything directly in the Streamlit sidebar.

### Supported Models (Groq)
| Model | Speed | Quality | Best For |
|-------|-------|---------|---------|
| `llama-3.3-70b-versatile` | Fast | Excellent | Default — best balance |
| `llama-3.1-8b-instant` | Ultra-fast | Good | Quick iterations |
| `mixtral-8x7b-32768` | Fast | Excellent | Long context tasks |
| `gemma2-9b-it` | Fast | Good | Efficient inference |

---

## Extending the System

### Add a New Specialist Agent
```python
# agents/specialist_agents.py
class CyberAgent:
    def __init__(self, api_key, model):
        self.llm = GroqClient(api_key, model)

    def run(self, state: AgentState) -> str:
        return self.llm.chat(CYBER_SYSTEM_PROMPT, ...)
```

```python
# agents/pipeline.py — add to route_to_specialists
Send("cyber_agent", state)
```

### Enable Real LangGraph
With `pip install langgraph langchain-groq`:
```python
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

llm = ChatGroq(api_key=api_key, model=model)
# ... build full graph as shown in Architecture section
```

### Add Reinforcement Learning
The MCTS reward function can be replaced with a trained policy:
```python
# agents/mcts_optimizer.py
def rollout(node, policy_network):
    # Use trained RL policy instead of LLM
    return policy_network.predict(node.state)
```

---

## Preset Missions
| Mission | Type | Complexity |
|---------|------|-----------|
| 🔴 Urban Assault — Seize City Hall | Direct Action | High |
| 🟡 Recon — Enemy Supply Depot | ISR | Medium |
| 🔵 Hostage Rescue — Mountain Compound | COIN/SF | Critical |
| ⚪ Air Superiority — Sector 9 | Air Operations | High |
| 🟠 Bridge Seizure — River Crossing | Air Assault | High |

---

## License
MIT License — for educational and research purposes.

> ⚠️ This system is a simulation tool for learning AI/ML concepts (multi-agent systems, HTN planning, MCTS). It does not reflect real military doctrine or classified information.
