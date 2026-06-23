"""
LangGraph Pipeline Orchestrator

This module implements the full LangGraph StateGraph pipeline.
It mirrors the LangGraph architecture with:
  - Sequential edges between planner → HTN → MCTS → simulator
  - Parallel Send() for specialist agents
  - Shared AgentState passed through all nodes

In a full LangGraph installation, this would use:
    from langgraph.graph import StateGraph, START, END, Send
    graph = StateGraph(AgentState)
    graph.add_node(...)
    graph.add_edge(...)
    app = graph.compile()
    result = await app.ainvoke(state)

Since we run without langgraph installed (for portability), we implement
the same flow imperatively with the same state semantics.
"""
import concurrent.futures
import time
from typing import Callable, Optional

from utils.state import AgentState
from agents.mission_planner import MissionPlannerAgent
from agents.htn_decomposer import HTNDecomposer
from agents.specialist_agents import ReconAgent, FireSupportAgent, LogisticsAgent
from agents.mcts_optimizer import MCTSOptimizer
from agents.execution_simulator import ExecutionSimulator


def run_pipeline(
    mission_brief: str,
    api_key: str,
    model: str,
    mcts_iterations: int = 40,
    mcts_C: float = 1.414,
    progress_cb: Optional[Callable] = None,
    agent_cb: Optional[Callable] = None,
) -> dict:
    """
    Execute the full multi-agent LangGraph pipeline.

    Graph topology:
        START
          ↓
        mission_planner_node
          ↓
        htn_decomposer_node
          ↓  [conditional edges via Send() — parallel fan-out]
         ├→ recon_agent_node        ─┐
         ├→ fire_support_agent_node  ├→ (join) → mcts_optimizer_node
         └→ logistics_agent_node   ─┘                ↓
                                          execution_simulator_node
                                                      ↓
                                                     END

    Returns:
        dict with keys:
            outputs: dict of agent outputs for UI display
            metrics: dict with tasks, iters, score, verdict
    """

    def log(msg, level="default"):
        if progress_cb:
            pass  # progress_cb handles both progress and logging
        print(f"[PIPELINE] {msg}")

    def prog(pct, msg, level="default"):
        if progress_cb:
            progress_cb(pct, msg, level)

    def set_agent(key, status):
        if agent_cb:
            agent_cb(key, status)

    # Initialize shared state
    state = AgentState(mission_brief=mission_brief)

    # Initialize all agents
    planner    = MissionPlannerAgent(api_key=api_key, model=model)
    htn        = HTNDecomposer(api_key=api_key, model=model)
    recon      = ReconAgent(api_key=api_key, model=model)
    fire       = FireSupportAgent(api_key=api_key, model=model)
    logistics  = LogisticsAgent(api_key=api_key, model=model)
    mcts       = MCTSOptimizer(api_key=api_key, model=model, iterations=mcts_iterations, C=mcts_C)
    simulator  = ExecutionSimulator(api_key=api_key, model=model)

    outputs = {}
    errors = []

    # ── NODE 1: Mission Planner ─────────────────────────────────────────────
    prog(5, "Mission Planner Agent → ACTIVE", "success")
    set_agent("mission_planner", "active")
    try:
        result = planner.run(state)
        state.planner_output = result
        outputs["planner"] = result
        set_agent("mission_planner", "done")
        prog(15, "Mission state extracted", "success")
    except Exception as e:
        errors.append(f"MissionPlanner: {e}")
        set_agent("mission_planner", "error")
        prog(15, f"Mission Planner error: {e}", "error")
        outputs["planner"] = f"ERROR: {e}"

    # ── NODE 2: HTN Decomposer ──────────────────────────────────────────────
    prog(18, "HTN Decomposer → ACTIVE", "success")
    set_agent("htn_decomposer", "active")
    try:
        result = htn.run(state)
        state.htn_output = result
        outputs["htn"] = result
        set_agent("htn_decomposer", "done")
        prog(32, f"HTN decomposed into {state.task_count or '?'} tasks", "success")
    except Exception as e:
        errors.append(f"HTNDecomposer: {e}")
        set_agent("htn_decomposer", "error")
        prog(32, f"HTN error: {e}", "error")
        outputs["htn"] = f"ERROR: {e}"

    # ── NODE 3: Parallel Specialist Agents (LangGraph Send() pattern) ───────
    prog(35, "Dispatching specialist agents in parallel (Send() API)...", "info")
    set_agent("recon_agent", "active")
    set_agent("fire_support_agent", "active")
    set_agent("logistics_agent", "active")

    def run_recon():
        return ("recon", recon.run(state))

    def run_fire():
        return ("fire", fire.run(state))

    def run_logistics():
        return ("logistics", logistics.run(state))

    try:
        # Parallel execution mirrors LangGraph's Send() fan-out
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_recon),
                executor.submit(run_fire),
                executor.submit(run_logistics),
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    key, result = future.result()
                    outputs[key] = result
                    agent_map = {"recon": "recon_agent", "fire": "fire_support_agent", "logistics": "logistics_agent"}
                    set_agent(agent_map[key], "done")
                    prog(45 + (5 if key == "recon" else 8 if key == "fire" else 11),
                         f"{key.replace('_', ' ').title()} Agent → Complete", "success")
                except Exception as e:
                    errors.append(f"Specialist {e}")
    except Exception as e:
        errors.append(f"Parallel agents: {e}")
        prog(58, f"Specialist agents error: {e}", "error")

    prog(58, "All specialist agents reported", "success")

    # ── NODE 4: MCTS Optimizer ─────────────────────────────────────────────
    prog(60, f"MCTS Optimizer → ACTIVE ({mcts_iterations} iterations)...", "success")
    set_agent("mcts_optimizer", "active")
    try:
        result = mcts.run(state)
        state.mcts_output = result
        outputs["mcts"] = result
        set_agent("mcts_optimizer", "done")
        prog(78, f"MCTS complete — score: {state.mcts_score:.3f}", "success")
    except Exception as e:
        errors.append(f"MCTS: {e}")
        set_agent("mcts_optimizer", "error")
        prog(78, f"MCTS error: {e}", "error")
        outputs["mcts"] = f"ERROR: {e}"

    # ── NODE 5: Execution Simulator ────────────────────────────────────────
    prog(80, "Execution Simulator → ACTIVE...", "success")
    set_agent("execution_simulator", "active")
    try:
        result = simulator.run(state)
        state.simulation_output = result
        outputs["simulation"] = result
        set_agent("execution_simulator", "done")
        prog(98, f"Simulation complete — Verdict: {state.verdict}", "success")
    except Exception as e:
        errors.append(f"Simulator: {e}")
        set_agent("execution_simulator", "error")
        prog(98, f"Simulator error: {e}", "error")
        outputs["simulation"] = f"ERROR: {e}"

    # ── Pipeline Complete ──────────────────────────────────────────────────
    prog(100, "Pipeline END — all nodes complete", "system")

    # Build summary
    verdict = state.verdict
    vmap = {"GO": "✓ GO", "NO-GO": "✗ NO-GO", "MODIFY": "◐ MODIFY"}
    outputs["summary"] = (
        f"PIPELINE COMPLETE\n"
        f"{'─'*42}\n"
        f"Model:      {model}\n"
        f"MCTS Iters: {mcts_iterations} (C={mcts_C})\n"
        f"HTN Tasks:  {state.task_count or '—'}\n"
        f"MCTS Score: {state.mcts_score:.3f}\n"
        f"Verdict:    {vmap.get(verdict, verdict)}\n"
        f"{'─'*42}\n"
        f"LangGraph nodes executed: 7\n"
        f"  START → mission_planner → htn_decomposer\n"
        f"       → [recon | fire_support | logistics] (parallel)\n"
        f"       → mcts_optimizer → execution_simulator → END"
    )

    if errors:
        outputs["summary"] += f"\n\nERRORS ENCOUNTERED:\n" + "\n".join(f"  • {e}" for e in errors)

    metrics = {
        "tasks": str(state.task_count) if state.task_count else "—",
        "iters": str(mcts_iterations),
        "score": f"{state.mcts_score:.2f}" if state.mcts_score else "—",
        "verdict": state.verdict or "—",
    }

    return {"outputs": outputs, "metrics": metrics, "state": state}
