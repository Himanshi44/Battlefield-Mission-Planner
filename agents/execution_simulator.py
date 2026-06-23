"""
Execution Simulator Agent — LangGraph Node 5 (Final)

Simulates the MCTS-optimized plan step-by-step.
For each action: estimates success probability, casualties,
resources consumed, and world state changes.

LangGraph role:
    graph.add_node("execution_simulator", self.run)
    graph.add_edge("mcts_optimizer", "execution_simulator")
    graph.add_edge("execution_simulator", END)

Output: SimulationResult (Pydantic model in full implementation)
"""
from utils.groq_client import GroqClient
from utils.state import AgentState

SYSTEM_PROMPT = """You are the Execution Simulator in a LangGraph multi-agent
military planning system powered by Groq.

Your role: Simulate the MCTS-optimized mission plan step-by-step.
For each action in the plan, evaluate:
- What actually happens (probabilistic outcome)
- Success / Partial Success / Failure status
- Casualties and resource consumption
- How the world state changes
- Decision points for contingencies

Output the COMPLETE EXECUTION SIMULATION in this format:

EXECUTION SIMULATION REPORT
══════════════════════════════════════════════

SIMULATION PARAMETERS:
- Monte Carlo Runs: 1000
- Random Seed: [any number]
- Confidence Interval: 95%
- Model: State-Space Simulation + LLM Rollout Policy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP-BY-STEP EXECUTION:

[STEP 1] Action: [action name from MCTS plan]
  ⟶ Status: SUCCESS ✓ / PARTIAL ◐ / FAILURE ✗
  ⟶ Probability: [%]
  ⟶ Casualties: Friendly [0] / Enemy [estimate]
  ⟶ Resources: [ammo/fuel/time consumed]
  ⟶ Duration: [time]
  ⟶ World State: [what changed — enemy status, terrain control, unit positions]
  ⟶ Notes: [any complications or lucky breaks]

[STEP 2] Action: [action name]
  ⟶ Status: [status]
  ⟶ Probability: [%]
  ⟶ Casualties: Friendly [0] / Enemy [estimate]
  ⟶ Resources: [consumed]
  ⟶ Duration: [time]
  ⟶ World State: [delta]
  ⟶ Notes: [notes]

[STEP 3] Action: [action name]
  ⟶ Status: [status]
  ⟶ Probability: [%]
  ⟶ Casualties: Friendly [0] / Enemy [estimate]
  ⟶ Resources: [consumed]
  ⟶ Duration: [time]
  ⟶ World State: [delta]
  ⟶ Notes: [notes]

[STEP 4] Action: [action name]
  ⟶ Status: [status]
  ⟶ Probability: [%]
  ⟶ Casualties: Friendly [0] / Enemy [estimate]
  ⟶ Resources: [consumed]
  ⟶ Duration: [time]
  ⟶ World State: [delta]
  ⟶ Notes: [notes]

[STEP 5] Action: [action name]
  ⟶ Status: [status]
  ⟶ Probability: [%]
  ⟶ Casualties: Friendly [0] / Enemy [estimate]
  ⟶ Resources: [consumed]
  ⟶ Duration: [time]
  ⟶ World State: [delta]
  ⟶ Notes: [notes]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SIMULATION SUMMARY:
┌─────────────────────────────────────────┐
│ MISSION SUCCESS PROBABILITY:  [%]        │
│ OBJECTIVES ACHIEVED:          [X/Y]      │
│ TOTAL FRIENDLY CASUALTIES:    [range]    │
│ TOTAL ENEMY NEUTRALIZED:      [estimate] │
│ TOTAL DURATION:               [H:MM]     │
│ RESOURCES CONSUMED:           [summary]  │
│ MISSION SCORE:                [0.0-1.0]  │
└─────────────────────────────────────────┘

CRITICAL DECISION POINTS IDENTIFIED:
1. [Decision point 1: condition → action required]
2. [Decision point 2: condition → action required]

CONTINGENCY TRIGGERS:
- If [event]: Execute [CONTINGENCY A]
- If [event]: Execute [CONTINGENCY B]
- If [event]: ABORT — extract forces via [route]

LESSONS LEARNED (AFTER-ACTION PREVIEW):
+ [What went well in simulation]
+ [Strength of plan]
- [Potential weakness identified]
- [Risk that could not be mitigated]

FINAL VERDICT:
╔══════════════════════════════════════╗
║  GO ✓ / NO-GO ✗ / MODIFY ◐          ║
║  [Brief justification in 1 sentence] ║
╚══════════════════════════════════════╝

Use realistic military simulation. Be specific about numbers and timelines."""


class ExecutionSimulator:
    """
    Node 5 (Final) of the LangGraph pipeline.

    Simulates the MCTS-optimized plan by:
    1. Parsing the optimal action sequence from MCTS output
    2. Calling the LLM to simulate each step (world state transitions)
    3. Computing aggregate mission metrics
    4. Issuing a GO/NO-GO/MODIFY verdict

    In full LangGraph implementation would use Pydantic structured outputs:
        class SimResult(BaseModel):
            step: int
            action: str
            status: Literal["SUCCESS", "PARTIAL", "FAILURE"]
            success_probability: float
            friendly_casualties: int
            enemy_casualties: int
            resources_consumed: dict[str, float]
            world_state_delta: str

        llm.with_structured_output(list[SimResult]).ainvoke(...)
    """

    def __init__(self, api_key: str, model: str):
        self.llm = GroqClient(api_key=api_key, model=model)
        self.name = "Execution Simulator"

    def run(self, state: AgentState) -> str:
        """Execute simulation of the optimal plan."""
        user_prompt = f"""Simulate the execution of this mission plan step-by-step.

MISSION BRIEF:
{state.mission_brief}

MCTS OPTIMAL PLAN:
{state.mcts_output[:1500] if state.mcts_output else 'N/A'}

RECON INTELLIGENCE:
{state.recon_report[:400] if state.recon_report else 'N/A'}

FIRE SUPPORT PLAN:
{state.fire_support_report[:400] if state.fire_support_report else 'N/A'}

LOGISTICS ESTIMATE:
{state.logistics_report[:400] if state.logistics_report else 'N/A'}

Simulate each key action in the optimal plan (5-7 steps minimum).
Be realistic about probabilities, casualties, and timelines.
Extract the 5-7 most critical actions from the MCTS plan and simulate each."""

        result = self.llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=2000,
            temperature=0.35,
        )

        # Parse verdict
        import re
        m = re.search(r"FINAL VERDICT[:\s]+.*?(GO|NO-GO|MODIFY)", result, re.DOTALL)
        if m:
            v = m.group(1)
            state.verdict = v

        # Parse success probability
        m2 = re.search(r"MISSION SUCCESS PROBABILITY:\s*(\d+)", result)
        if m2:
            state.success_probability = int(m2.group(1)) / 100.0

        state.simulation_output = result
        return result
