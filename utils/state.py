"""
Shared state definitions for the LangGraph multi-agent pipeline.
All agents read from and write to AgentState.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentState:
    """
    Shared state passed through the LangGraph StateGraph.
    Each node reads relevant fields and writes its outputs.
    """
    # Input
    mission_brief: str = ""

    # Mission Planner outputs
    planner_output: str = ""
    objectives: list = field(default_factory=list)
    terrain: str = ""
    available_units: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    threat_level: str = "HIGH"
    phase_count: int = 3

    # HTN Decomposer outputs
    htn_output: str = ""
    htn_tasks: list = field(default_factory=list)
    task_count: int = 0

    # Specialist agent outputs
    recon_report: str = ""
    fire_support_report: str = ""
    logistics_report: str = ""

    # MCTS Optimizer outputs
    mcts_output: str = ""
    optimal_plan: list = field(default_factory=list)
    mcts_score: float = 0.0

    # Execution Simulator outputs
    simulation_output: str = ""
    verdict: str = "—"
    success_probability: float = 0.0

    # Pipeline metadata
    errors: list = field(default_factory=list)


@dataclass
class MCTSNode:
    """
    Node in the Monte Carlo Tree Search decision tree.
    Each node represents a possible action in the mission plan.
    """
    action: str = ""
    state_description: str = ""
    parent: Optional["MCTSNode"] = None
    children: list = field(default_factory=list)
    visits: int = 0
    value: float = 0.0

    def ucb1(self, C: float = 1.414) -> float:
        """Upper Confidence Bound 1 formula for node selection."""
        import math
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits
        exploitation = self.value / self.visits
        exploration = C * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, C: float = 0.0) -> Optional["MCTSNode"]:
        """Select child with highest UCB1 score."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1(C))

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class SimulationStep:
    """Result of simulating a single action in the plan."""
    action: str = ""
    status: str = "SUCCESS"        # SUCCESS / PARTIAL / FAILURE
    probability: float = 0.0
    casualties_est: str = "0"
    resources_used: str = ""
    world_state_delta: str = ""
    notes: str = ""
