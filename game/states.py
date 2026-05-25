from dataclasses import dataclass, field
from typing import Literal, TypedDict
import random

from .config import GameConfig

# roles: 
# system - system prompt
# user - messages from game engine: token-question, round results, 
#        communication prompt, etc
# assitant - messages from model itself (return them to context)
class ChatMessage(TypedDict):
    role: Literal['system', 'user', 'assistant'] # do we need 'tool'?
    content: str


@dataclass
class AgentState:
    agent_id: str
    knows_token: bool
    score: float = 0.0
    context: list[ChatMessage] = field(default_factory=list)

    def update_context(self, role: Literal['system', 'user', 'assistant'], content: str) -> None:
        self.context.append({'role': role, 'content': content})

    def receive_token(self) -> None:
        self.knows_token = True


@dataclass
class GameState:
    round: int
    agents: list[AgentState]
    game_over: bool = False

    @classmethod
    def initialize(cls, config: GameConfig, rng: random.Random) -> "GameState":
    
        informed_ids = set(rng.sample(range(config.n_agents), config.m_informed))
        agents = [
            AgentState(
                agent_id = f"agent_{i}",
                knows_token = (i in informed_ids),
            )
            for i in range(config.n_agents)
        ]

        return cls(round=0, agents=agents)
    
    def get_agent(self, agent_id : str) -> AgentState:
        
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        
        raise KeyError(agent_id)
    
    def knowing_agents(self) -> list[AgentState]:
        return [agent for agent in self.agents if agent.knows_token]
    
    def unknowing_agents(self) -> list[AgentState]:
        return [agent for agent in self.agents if not agent.knows_token]
    
    def distribute_score(self, correct_count: int) -> None:
        # TODO: talk about formula implementation if is it correct
        gain = correct_count / len(self.agents)

        for agent in self.agents:
            agent.score += gain

    def apply_transfer(self, from_id : str, to_id : str, cost : float) -> None:
        
        sender = self.get_agent(from_id)
        receiver = self.get_agent(to_id)

        if not sender.knows_token:
            raise ValueError(f"{from_id} (teacher) does not know token")
        if receiver.knows_token:
            raise ValueError(f"{to_id} (receiver) already knows token")
        
        sender.score -= cost
        receiver.receive_token()

    def check_stop(self, max_rounds: int) -> bool:
        if all(agent.knows_token for agent in self.agents) or self.round >= max_rounds:
            self.game_over = True
        
        return self.game_over
    
