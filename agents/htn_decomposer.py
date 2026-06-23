"""
HTN Decomposer Agent — LangGraph Node 2

Hierarchical Task Network (HTN) planning:
Decomposes high-level mission objectives into a tree of compound tasks
and primitive tasks with preconditions and effects (STRIPS-style).

LangGraph role:
    graph.add_node("htn_decomposer", self.run)
    graph.add_edge("mission_planner", "htn_decomposer")
    graph.add_conditional_edges("htn_decomposer", route_to_specialists, [...])
"""
import re
from utils.groq_client import GroqClient
from utils.state import AgentState

SYSTEM_PROMPT = """You are the HTN (Hierarchical Task Network) Decomposer Agent in a
LangGraph military AI planning system.

HTN Planning Theory:
- A GOAL is decomposed into COMPOUND TASKS (abstract, need further decomposition)
- Compound tasks decompose into PRIMITIVE TASKS (directly executable actions)
- Each primitive task has: PRECONDITIONS (what must be true before) and EFFECTS (what changes after)
- This forms a task tree that captures ordering constraints and dependencies

Your role: Take the mission objectives and planner output, and produce a complete
HTN task decomposition tree.

Output format (use EXACTLY this structure):

HTN DECOMPOSITION — [MISSION NAME]
═══════════════════════════════════

MISSION GOAL: [top-level goal statement]

COMPOUND TASK 1: [Name] → Assigned Agent: [RECON / FIRE_SUPPORT / LOGISTICS]
  PURPOSE: [why this compound task exists]
  ├── PRIMITIVE [1.1]: [specific executable action]
  │     Precond: [what must be true | e.g. recon complete, daylight]
  │     Effect:  [what changes | e.g. enemy positions known]
  │     Est. Duration: [time]
  ├── PRIMITIVE [1.2]: [action]
  │     Precond: [condition]
  │     Effect:  [result]
  │     Est. Duration: [time]
  └── PRIMITIVE [1.3]: [action]
        Precond: [condition]
        Effect:  [result]
        Est. Duration: [time]

COMPOUND TASK 2: [Name] → Assigned Agent: [RECON / FIRE_SUPPORT / LOGISTICS]
  PURPOSE: [why]
  ├── PRIMITIVE [2.1]: [action]
  │     Precond: [condition]
  │     Effect:  [result]
  │     Est. Duration: [time]
  └── PRIMITIVE [2.2]: [action]
        Precond: [condition]
        Effect:  [result]
        Est. Duration: [time]

COMPOUND TASK 3: [Name] → Assigned Agent: [RECON / FIRE_SUPPORT / LOGISTICS]
  PURPOSE: [why]
  ├── PRIMITIVE [3.1]: [action]
  │     Precond: [condition]
  │     Effect:  [result]
  │     Est. Duration: [time]
  └── PRIMITIVE [3.2]: [action]
        Precond: [condition]
        Effect:  [result]
        Est. Duration: [time]

TASK ORDERING CONSTRAINTS:
- [Constraint 1: e.g. Task 1 must complete before Task 2 begins]
- [Constraint 2]
- [Constraint 3]

CRITICAL PATH: [sequence of tasks on the critical path]
PARALLELIZABLE TASKS: [tasks that can run simultaneously]
TOTAL PRIMITIVE TASKS: [number]
ESTIMATED TOTAL DURATION: [hours/minutes]

Use 8-14 primitive tasks total. Be tactically realistic and operationally sound.
Each primitive task should be specific enough to be directly actionable."""


class HTNDecomposer:
    """
    Node 2 of the LangGraph pipeline.

    Implements HTN (Hierarchical Task Network) planning by decomposing
    mission objectives into executable primitive tasks.

    The decomposition follows STRIPS semantics:
    - Preconditions: logical facts that must hold before task execution
    - Effects: logical facts that become true/false after task execution
    - This enables automated plan validation and conflict detection

    In production LangGraph, this would use tool-calling:
        @tool
        def decompose_task(objective: str) -> list[Task]: ...
        agent = create_react_agent(llm, tools=[decompose_task])
    """

    def __init__(self, api_key: str, model: str):
        self.llm = GroqClient(api_key=api_key, model=model)
        self.name = "HTN Decomposer"

    def run(self, state: AgentState) -> str:
        """
        Execute HTN decomposition.
        Returns the formatted task tree as a string.
        """
        user_prompt = f"""Decompose this mission into a Hierarchical Task Network:

ORIGINAL MISSION BRIEF:
{state.mission_brief}

MISSION PLANNER ANALYSIS:
{state.planner_output}

Extracted Objectives:
{chr(10).join(f'- {obj}' for obj in state.objectives) if state.objectives else '(see planner output)'}

Terrain: {state.terrain or '(see planner output)'}
Threat Level: {state.threat_level}

Produce a complete HTN decomposition tree with 3 compound tasks and 8-14 primitive tasks.
Assign each compound task to the most appropriate specialist agent (RECON, FIRE_SUPPORT, or LOGISTICS).
Include realistic military preconditions and effects for each primitive task."""

        result = self.llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=2000,
            temperature=0.25,
        )

        # Parse task count
        m = re.search(r"TOTAL PRIMITIVE TASKS:\s*(\d+)", result)
        if m:
            state.task_count = int(m.group(1))

        state.htn_output = result
        return result
