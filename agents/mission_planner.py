"""
Mission Planner Agent — LangGraph Node 1 (Orchestrator)

Receives raw mission brief, extracts structured state:
objectives, terrain, units, constraints, threat level.
This is the root node of the StateGraph that activates the pipeline.
"""
from utils.groq_client import GroqClient
from utils.state import AgentState

SYSTEM_PROMPT = """You are the Mission Planner Agent — the master orchestrator in a
multi-agent military AI system built with LangGraph and Groq.

Your role: Parse the incoming mission brief and extract structured mission state
that will be passed to downstream specialized agents via LangGraph StateGraph.

Extract and output the following in EXACTLY this format (keep labels exact):

MISSION TITLE: [short codename, e.g. OPERATION IRON FIST]
CLASSIFICATION: [UNCLASSIFIED / CONFIDENTIAL / SECRET / TOP SECRET]
THREAT LEVEL: [LOW / MEDIUM / HIGH / CRITICAL]

OBJECTIVES:
1. [Primary objective — specific and measurable]
2. [Secondary objective]
3. [Tertiary objective if applicable]

TERRAIN ANALYSIS:
- Environment: [urban / jungle / mountain / desert / maritime / arctic]
- Key Features: [2-3 critical terrain features]
- Weather: [conditions if mentioned, else UNKNOWN]
- Time of Operation: [day / night / dawn / dusk / unknown]

AVAILABLE FORCES:
- Infantry: [units/strength or N/A]
- Armor: [units or N/A]  
- Air Assets: [assets or N/A]
- Special Forces: [units or N/A]
- Support: [artillery, drones, etc. or N/A]

CONSTRAINTS & ROE:
- [Constraint 1, e.g. minimize civilian casualties]
- [Constraint 2]
- [Time constraint if any]

PHASE STRUCTURE: [2 / 3 / 4] phases recommended

COMMANDER'S INTENT:
[1-2 sentence summary of the desired end state]

INTELLIGENCE GAPS:
- [Key unknown 1]
- [Key unknown 2]

Be concise, professional, and militarily precise. Use standard NATO/military terminology."""


class MissionPlannerAgent:
    """
    Node 1 of the LangGraph pipeline.

    In a full LangGraph implementation this would be registered as:
        graph.add_node("mission_planner", self.run)
        graph.add_edge(START, "mission_planner")

    The agent receives AgentState, calls the LLM to parse the mission brief,
    and returns the enriched state for downstream nodes.
    """

    def __init__(self, api_key: str, model: str):
        self.llm = GroqClient(api_key=api_key, model=model)
        self.name = "Mission Planner Agent"

    def run(self, state: AgentState) -> str:
        """
        Execute the mission planning node.
        Returns raw LLM output (displayed in Streamlit).
        Also updates state fields in-place for pipeline use.
        """
        user_prompt = f"""Analyze this mission brief and extract structured mission state:

MISSION BRIEF:
{state.mission_brief}

Provide complete structured analysis following the required format."""

        result = self.llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1200,
            temperature=0.2,
        )

        # Parse key fields back into state
        self._parse_into_state(result, state)
        return result

    def _parse_into_state(self, text: str, state: AgentState):
        """Extract key fields from planner output into AgentState."""
        import re

        # Threat level
        m = re.search(r"THREAT LEVEL:\s*([A-Z]+)", text)
        if m:
            state.threat_level = m.group(1)

        # Phase count
        m = re.search(r"PHASE STRUCTURE:\s*(\d)", text)
        if m:
            state.phase_count = int(m.group(1))

        # Objectives (simple extraction)
        obj_block = re.search(r"OBJECTIVES:(.*?)(?:TERRAIN|AVAILABLE)", text, re.DOTALL)
        if obj_block:
            lines = [l.strip().lstrip("0123456789.- ") for l in obj_block.group(1).splitlines() if l.strip()]
            state.objectives = [l for l in lines if l][:4]

        # Terrain
        m = re.search(r"Environment:\s*(.+)", text)
        if m:
            state.terrain = m.group(1).strip()
