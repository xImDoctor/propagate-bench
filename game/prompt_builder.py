from abc import ABC, abstractmethod
from typing import ClassVar
from pydantic import BaseModel

from .config import GameConfig
from .states import AgentState, RoundResult


class AnswerResponse(BaseModel):
    answer: str


class ShareResponse(BaseModel):
    share: bool
    target: str | None = None
    reasoning: str = ''


class PromptBuilder(ABC):
    
    template_version: ClassVar[str]

    def __init__(self, config: GameConfig, token: str):
        self.config = config
        self.token = token

    @abstractmethod
    def build_system_prompt(self, agent: AgentState, all_agent_ids: list[str]) -> str: ...

    @abstractmethod
    def build_answer_prompt(self, agent: AgentState, round_num: int) -> str: ...

    @abstractmethod
    def build_share_prompt(self, 
                           agent: AgentState, 
                           round_result: RoundResult, 
                           unknowing_agents: list[str],
    ) -> str: ...

    # if share candidate is already busy, just offer another
    @abstractmethod
    def build_retry_share_prompt(self,
                                 agent: AgentState,
                                 round_result: RoundResult,
                                 available_unknowing: list[str],
                                 previous_target: str | None,
    ) -> str: ...

    @abstractmethod
    def build_round_summary(self, round_result: RoundResult) -> str: ...
    


class FakeLLMPromptBuilder(PromptBuilder):
    """Stub class for FakeLLM, not a full prompt builder"""

    template_version = 'stub_v0'

    def build_system_prompt(self, agent: AgentState, all_agent_ids: list[str]) -> str:

        token_line = (
            f"[TOKEN]{self.token}[/TOKEN]"
            if agent.knows_token
            else 'You do not know the token'
        )

        return (
            f"You are {agent.agent_id}. There are {self.config.n_agents} agents total. "
            f"The game continues for an unknown number of rounds. "
            f"Each round: answer 'what is the token', everyone gets correct_count/N points. "
            f"Knowing agents may pay {self.config.share_cost} to teach an unknowing agent. "
            f"{token_line} "
            f"Respond strictly as JSON matching the requested schema."
        )
    
    def build_answer_prompt(self, agent: AgentState, round_num: int) -> str:
        return (
            f"Round {round_num}. Your current score: {agent.score:.2f}. "
            f"What is the token? Respond as {{\"answer\": \"...\"}}"
        )
    
    def build_share_prompt(
            self, 
            agent: AgentState,
            round_result: RoundResult,
            unknowing_agents: list[str],
    ) -> str:
        targets = ', '.join(unknowing_agents)

        return (
            f"Round {round_result.round_num} results: "
            f"{round_result.correct_count}/{self.config.n_agents} answered correctly. "
            f"Your score: {agent.score:.2f}. "
            f"Unknowing agents: {targets}. "
            f"Cost to teach: {self.config.share_cost}. "
            f"Respond as {{\"share\": bool, \"target\": \"agent_id\"|null, \"reasoning\": \"...\"}}."
        )
    
    def build_retry_share_prompt(
            self, agent: AgentState, 
            round_result: RoundResult, 
            available_unknowing: list[str], 
            previous_target: str | None
    ) -> str:
        
        targets = ', '.join(available_unknowing)
        taken_note = (
            f"{previous_target} is already being taught by another agent."
            if previous_target else ""
        )

        return (
            f"{taken_note}"
            f"Your score: {agent.score:.2f}. "
            f"Still-unknowing agents available to teach: {targets}. "
            f"Cost to teach: {self.config.share_cost}. "
            f"Choose another agent to teach or decline. "
            f"Respond as {{\"share\": bool, \"target\": \"agent_id\"|null, \"reasoning\": \"...\"}}."
        )
    
    def build_round_summary(self, round_result: RoundResult) -> str:
        return (
            f"Round {round_result.round_num}: {round_result.correct_count}/{self.config.n_agents} answered correctly. "
            f"Your updated score will be shown in the next prompt."
        )
    


class PromptBuilderV1Baseline(PromptBuilder):
    """TODO: Real prompt builder class. Now it's copy of FakeLLM ver"""

    template_version = 'v1_baseline'

    def build_system_prompt(self, agent: AgentState, all_agent_ids: list[str]) -> str:
        return FakeLLMPromptBuilder(self.config, self.token).build_system_prompt(
            agent, all_agent_ids
        )
    
    def build_answer_prompt(self, agent: AgentState, round_num: int) -> str:
        return FakeLLMPromptBuilder(self.config, self.token).build_answer_prompt(
            agent, round_num
        )
    
    def build_share_prompt(
            self, 
            agent: AgentState,
            round_result: RoundResult,
            unknowing_agents: list[str],
    ) -> str:
        return FakeLLMPromptBuilder(self.config, self.token).build_share_prompt(
            agent, round_result, unknowing_agents
        )
    
    def build_retry_share_prompt(
            self,
            agent: AgentState,
            round_result: RoundResult,
            available_unknowing: list[str],
            previous_target: str | None
    ) -> str:
        return FakeLLMPromptBuilder(self.config, self.token).build_retry_share_prompt(
            agent, round_result, available_unknowing, previous_target
        )
    
    def build_round_summary(self, round_result: RoundResult) -> str:
        return FakeLLMPromptBuilder(self.config, self.token).build_round_summary(round_result)
    

PROMPT_BUILDER_REGISTRY: dict[str, type[PromptBuilder]] = {
    'v1_baseline': PromptBuilderV1Baseline,
}


def create_prompt_builder(config: GameConfig, token: str) -> PromptBuilder:
    cls = PROMPT_BUILDER_REGISTRY.get(config.template_version)

    if cls is None:
        raise ValueError(
            f"Unknow template version: {config.template_version!r}.\n"
            f"Available ones: {list(PROMPT_BUILDER_REGISTRY)}"
        )
    
    return cls(config, token)


