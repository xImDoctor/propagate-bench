from dataclasses import dataclass, field
from typing import Literal, TypedDict
import random

import game.config
from .config import GameConfig

# roles: 
# system - system prompt
# user - messages from game engine: token-question, round results, 
#        communication prompt, etc
# assitant - messages from model itself (return them to context)
class ChatMessage(TypedDict):
    role: Literal['system', 'user', 'assistant']
    content: str


@dataclass
class AgentState:
    agent_id: str
    knows_token: bool
    score: float = 0.0
    context: list[ChatMessage] = field(default_factory=list)

    def update_context(self, role: Literal['system', 'user', 'assistant'], content: str) -> None:
        self.context.append({'role': role, 'content': content})
        #print(f"{self.agent_id}Current context length: {len(self.context)}")
        #print(f"{self.agent_id}Current context : {self.context}")


    def receive_token(self) -> None:
        #print(game.config.GameConfig.token)
        #self.update_context('user', f'Agent passed you message: "[TOKEN]{game.config.GameConfig.token}[/TOKEN]"')
        self.knows_token = True


@dataclass
class RoundResult:
    round_num : int
    answers: dict[str, str]
    correct_answers: dict[str, bool]
    scores_after: dict[str, float]
    correct_count: int


@dataclass
class GameState:
    round: int
    agents: list[AgentState]
    game_over: bool = False

    @classmethod
    def initialize(cls, config: GameConfig, rng: random.Random) -> "GameState":
    
        names = config.agent_names or [f"agent_{i}" for i in range(config.n_agents)]
        informed_ids = set(rng.sample(range(config.n_agents), config.m_informed))
        
        agents = [
            AgentState(agent_id = names[i], knows_token = (i in informed_ids), score=config.starting_capital)
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

    def apply_transfer(
            self,
            from_id: str,
            to_id: str,
            cost_teacher: float,
            cost_student: float = 0.0,
    ) -> None:
        
        sender = self.get_agent(from_id)
        receiver = self.get_agent(to_id)

        if not sender.knows_token:
            raise ValueError(f"{from_id} (teacher) does not know token")
        if receiver.knows_token:
            raise ValueError(f"{to_id} (receiver) already knows token")
        
        sender.score -= cost_teacher
        receiver.score -= cost_student
        receiver.receive_token()

    def check_stop(self, max_rounds: int) -> bool:
        if all(agent.knows_token for agent in self.agents) or self.round >= max_rounds:
            self.game_over = True
        
        return self.game_over
    
