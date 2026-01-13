import os

# Top-level shim package to make imports like `from agents.base_agent import ...`
# resolve to the real implementation under agentboard/agents.

__all__ = []

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_agentboard_agents = os.path.join(_root, "agentboard", "agents")
if os.path.isdir(_agentboard_agents):
    # Insert at front to prefer files under agentboard/agents
    if _agentboard_agents not in __path__:
        __path__.insert(0, _agentboard_agents)
