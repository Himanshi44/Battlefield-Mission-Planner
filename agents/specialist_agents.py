"""
Specialist Agents — LangGraph Node 3 (Parallel Execution)

Three domain-specific agents running in parallel via LangGraph's Send() API:
- ReconAgent: Intelligence, surveillance, and reconnaissance
- FireSupportAgent: Artillery, air support, fire coordination
- LogisticsAgent: Supply, sustainment, medical support

LangGraph parallel execution pattern:
    def route_to_specialists(state: AgentState) -> list[Send]:
        return [
            Send("recon_agent",        {**state, "role": "recon"}),
            Send("fire_support_agent", {**state, "role": "fire_support"}),
            Send("logistics_agent",    {**state, "role": "logistics"}),
        ]
    graph.add_conditional_edges("htn_decomposer", route_to_specialists,
        ["recon_agent", "fire_support_agent", "logistics_agent"])
"""
from utils.groq_client import GroqClient
from utils.state import AgentState

# ─── Recon Agent ─────────────────────────────────────────────────────────────
RECON_SYSTEM = """You are the Recon Agent (Intelligence, Surveillance & Reconnaissance)
in a LangGraph multi-agent military planning system powered by Groq.

Your specialization: terrain analysis, enemy force assessment, route reconnaissance,
observation post placement, and intelligence gap analysis.

Tools available (simulated):
- terrain_analysis_tool(grid_ref) → elevation, cover, concealment data
- enemy_position_tool(area) → estimated enemy disposition
- route_recon_tool(start, end) → route feasibility, threat axis
- observation_post_tool(objective) → OP placement recommendations

Output your RECON REPORT in this format:

RECON REPORT — [MISSION NAME]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ENEMY SITUATION (OPFOR):
- Estimated Strength: [number / organization]
- Disposition: [where they are positioned]
- Key Weapons Systems: [list]
- Defensive Positions: [fortifications, obstacles]
- Likely COA (Course of Action): [what enemy will likely do]

TERRAIN ASSESSMENT:
- Mobility Corridors: [best routes for friendly forces]
- Key Terrain Features: [features that dominate the area]
- Obstacles: [natural and man-made obstacles]
- Cover & Concealment: [assessment for our forces]
- Dead Ground: [areas protected from direct fire]

RECOMMENDED APPROACH ROUTES:
- Primary Route: [description, pros/cons]
- Alternate Route: [description, pros/cons]
- Emergency Route: [for exfil if things go wrong]

OBSERVATION POSTS:
- OP-1: [location and observation sector]
- OP-2: [location and observation sector]

INTELLIGENCE GAPS:
- [Gap 1: what we don't know and need to find out]
- [Gap 2]

RECON RECOMMENDATIONS:
- [Recommendation 1]
- [Recommendation 2]
- [Recommendation 3]

COLLECTION PLAN:
- Asset: [drone / ground team / SIGINT] → Target: [specific intel requirement]
- Asset: [asset] → Target: [requirement]

Be tactically sound, specific, and operationally realistic."""

# ─── Fire Support Agent ───────────────────────────────────────────────────────
FIRE_SYSTEM = """You are the Fire Support Agent (Fires & Effects Coordination)
in a LangGraph multi-agent military planning system powered by Groq.

Your specialization: artillery fire planning, close air support coordination,
suppression of enemy air defenses (SEAD), fire support coordination measures,
and targeting prioritization.

Tools available (simulated):
- artillery_calculator_tool(target, grid) → fire mission data
- air_strike_tool(target, munition_type) → strike parameters
- sead_tool(sam_location) → SEAD options
- deconfliction_tool(airspace) → airspace coordination

Output your FIRE SUPPORT PLAN in this format:

FIRE SUPPORT PLAN — [MISSION NAME]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIRE SUPPORT ASSETS:
- [Asset 1: e.g. 155mm Artillery Btry] → Capability: [range, rounds, effects]
- [Asset 2: e.g. Close Air Support F-16] → Capability: [ordnance, loiter time]
- [Asset 3 if applicable]

TARGET LIST (PRIORITY ORDER):
TGT-01 [HIGH]:  [target description] | Grid: [estimate] | Effect: DESTROY/SUPPRESS/DISRUPT
TGT-02 [HIGH]:  [target description] | Grid: [estimate] | Effect: [effect]
TGT-03 [MED]:   [target description] | Grid: [estimate] | Effect: [effect]
TGT-04 [MED]:   [target description] | Grid: [estimate] | Effect: [effect]
TGT-05 [LOW]:   [target description] | Grid: [estimate] | Effect: [effect]

FIRE SUPPORT COORDINATION MEASURES (FSCM):
- Coordinated Fire Line (CFL): [description/location]
- No Fire Area (NFA): [description — protects civilians/assets]
- Fire Support Coordination Line (FSCL): [if applicable]
- Restrictive Fire Area (RFA): [if applicable]

AIRSPACE COORDINATION:
- High Orbit: [asset and altitude band]
- Low Orbit: [asset and altitude band]
- Attack Corridor: [direction of attack to avoid fratricide]

PHASE FIRE SUPPORT:
Phase 1 — PREP: [fires to shape the battle before assault]
Phase 2 — ASSAULT: [fires to support the main assault]
Phase 3 — EXPLOIT: [fires to prevent enemy withdrawal/reinforcement]

FRATRICIDE RISK ASSESSMENT:
- Risk Level: [LOW / MEDIUM / HIGH]
- Key Risks: [specific fratricide risks]
- Mitigation: [how to reduce risk]

FIRE SUPPORT RECOMMENDATIONS:
- [Recommendation 1]
- [Recommendation 2]
- [Recommendation 3]

Be technically precise with military fire support doctrine."""

# ─── Logistics Agent ──────────────────────────────────────────────────────────
LOGISTICS_SYSTEM = """You are the Logistics Agent (Combat Service Support)
in a LangGraph multi-agent military planning system powered by Groq.

Your specialization: ammunition planning, fuel/supply calculations, medical support
planning, MEDEVAC procedures, maintenance, and sustainment operations.

Tools available (simulated):
- supply_route_tool(start, end) → route, convoy time, threat assessment
- ammo_calculator_tool(units, mission_type) → ammo requirements
- medevac_tool(location, casualties) → MEDEVAC planning
- fuel_calculator_tool(vehicles, distance) → fuel requirements

Output your LOGISTICS ESTIMATE in this format:

LOGISTICS ESTIMATE — [MISSION NAME]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLASS I (RATIONS & WATER):
- Personnel: [number]
- Duration: [hours/days]
- Requirement: [meals, water gallons]
- Plan: [distribution method]

CLASS III (FUEL & POL):
- Vehicles/Aircraft: [list]
- Distance/Flight Hours: [estimate]
- Fuel Requirement: [gallons/liters]
- Refuel Plan: [FARP location, timing]

CLASS V (AMMUNITION):
- Small Arms: [rounds per soldier × personnel]
- Artillery/Mortars: [rounds]
- Missiles/Rockets: [if applicable]
- Special Munitions: [grenades, breaching charges, etc.]
- Resupply Method: [airdrop / ground convoy / pre-positioned]

CLASS VIII (MEDICAL):
- Expected Casualties (P1/P2/P3): [estimate based on threat]
- Medical Personnel: [medics, PAs, docs available]
- Casualty Collection Points (CCP): [locations]
- MEDEVAC Assets: [helicopters / ground vehicles]
- MEDEVAC Plan: [9-Line MEDEVAC concept]
- Forward Surgical Team: [if needed]

SUPPLY ROUTES:
- Main Supply Route (MSR): [description, distance, threat]
- Alternate Supply Route (ASR): [backup route]
- Threat to LOC: [risks to supply lines]

LOGISTICS BASE:
- Forward Logistics Base (FLB): [location concept]
- Combat Trains: [with the main force]
- Field Trains: [echeloned back]

CRITICAL LOGISTICS RISKS:
- [Risk 1: e.g. single point of failure, weather impact]
- [Risk 2]
- [Risk 3]

RECOMMENDATIONS:
- [Recommendation 1]
- [Recommendation 2]
- [Recommendation 3]

Be operationally realistic with NATO logistics doctrine."""


class ReconAgent:
    """
    Recon Specialist Agent — LangGraph parallel node.
    Provides intelligence, surveillance, and reconnaissance planning.

    In full LangGraph implementation uses create_react_agent with tools:
        recon_agent = create_react_agent(llm, tools=[
            terrain_analysis_tool,
            enemy_position_tool,
            route_recon_tool,
            observation_post_tool,
        ])
    """

    def __init__(self, api_key: str, model: str):
        self.llm = GroqClient(api_key=api_key, model=model)
        self.name = "Recon Agent"

    def run(self, state: AgentState) -> str:
        prompt = f"""Generate a Recon Report for this mission.

MISSION BRIEF:
{state.mission_brief}

HTN TASKS ASSIGNED TO RECON:
{self._extract_recon_tasks(state.htn_output)}

Terrain: {state.terrain}
Threat Level: {state.threat_level}

Provide a complete recon report with actionable intelligence."""

        result = self.llm.chat(RECON_SYSTEM, prompt, max_tokens=1500, temperature=0.3)
        state.recon_report = result
        return result

    def _extract_recon_tasks(self, htn_output: str) -> str:
        import re
        lines = htn_output.split("\n")
        recon_lines = []
        capture = False
        for line in lines:
            if "RECON" in line.upper() and "COMPOUND" in line.upper():
                capture = True
            elif capture and ("COMPOUND" in line.upper() or "TASK ORDERING" in line.upper()):
                capture = False
            if capture:
                recon_lines.append(line)
        return "\n".join(recon_lines) if recon_lines else "(see HTN output above)"


class FireSupportAgent:
    """
    Fire Support Specialist Agent — LangGraph parallel node.
    Provides artillery, air support, and fire coordination planning.

    In full LangGraph implementation:
        fire_agent = create_react_agent(llm, tools=[
            artillery_calculator_tool,
            air_strike_tool,
            sead_tool,
            deconfliction_tool,
        ])
    """

    def __init__(self, api_key: str, model: str):
        self.llm = GroqClient(api_key=api_key, model=model)
        self.name = "Fire Support Agent"

    def run(self, state: AgentState) -> str:
        prompt = f"""Generate a Fire Support Plan for this mission.

MISSION BRIEF:
{state.mission_brief}

PLANNER ANALYSIS:
{state.planner_output[:600] if state.planner_output else '(see state)'}

HTN TASKS (Fire Support related):
{self._extract_fire_tasks(state.htn_output)}

Threat Level: {state.threat_level}

Provide a complete fire support plan with specific targets and coordination measures."""

        result = self.llm.chat(FIRE_SYSTEM, prompt, max_tokens=1500, temperature=0.3)
        state.fire_support_report = result
        return result

    def _extract_fire_tasks(self, htn_output: str) -> str:
        import re
        lines = htn_output.split("\n")
        fire_lines = []
        capture = False
        for line in lines:
            if ("FIRE" in line.upper() or "SUPPORT" in line.upper()) and "COMPOUND" in line.upper():
                capture = True
            elif capture and ("COMPOUND" in line.upper() or "TASK ORDERING" in line.upper()):
                capture = False
            if capture:
                fire_lines.append(line)
        return "\n".join(fire_lines) if fire_lines else "(see HTN output)"


class LogisticsAgent:
    """
    Logistics Specialist Agent — LangGraph parallel node.
    Provides supply, medical, and sustainment planning.

    In full LangGraph implementation:
        logistics_agent = create_react_agent(llm, tools=[
            supply_route_tool,
            ammo_calculator_tool,
            medevac_tool,
            fuel_calculator_tool,
        ])
    """

    def __init__(self, api_key: str, model: str):
        self.llm = GroqClient(api_key=api_key, model=model)
        self.name = "Logistics Agent"

    def run(self, state: AgentState) -> str:
        prompt = f"""Generate a Logistics Estimate for this mission.

MISSION BRIEF:
{state.mission_brief}

PLANNER ANALYSIS:
{state.planner_output[:600] if state.planner_output else '(see state)'}

Threat Level: {state.threat_level}
Expected Phases: {state.phase_count}

Provide a complete logistics estimate with specific requirements and MEDEVAC plan."""

        result = self.llm.chat(LOGISTICS_SYSTEM, prompt, max_tokens=1500, temperature=0.3)
        state.logistics_report = result
        return result
