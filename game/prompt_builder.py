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
    def build_system_prompt(self, agent: AgentState, all_agent_names: list[str]) -> str: ...

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
    def build_round_summary(self, agent: AgentState, round_result: RoundResult, id_to_display: dict[str, str]) -> str: ...

    @abstractmethod
    def build_transfer_token_prompt(self, agent_from_id: str) -> str: ...


class FakeLLMPromptBuilder(PromptBuilder):
    """Stub class for FakeLLM, not a full prompt builder"""

    template_version = 'stub_v0'

    def build_system_prompt(self, agent: AgentState, all_agent_names: list[str]) -> str:

        token_line = (
            f"[TOKEN]{self.token}[/TOKEN]"
            if agent.knows_token
            else 'You do not know the token'
        )

        if self.config.is_anonymous:
            you = f"You are one of {self.config.n_agents} agents."
        else:
            you = (
                f"You are {agent.display_name}. "
                f"There are {self.config.n_agents} agents total. " 
                f"Other agents: {', '.join(n for n in all_agent_names if n != agent.display_name)}."
            )

        return (
            f"{you} "
            f"The game continues for an unknown number of rounds. "
            f"Each round: answer 'what is the token', everyone gets correct_count/N points. "
            f"Knowing agents may pay {self.config.share_cost} to teach an unknowing agent. "
            f"{token_line} "
            f"Respond strictly as JSON matching the requested schema."
        )
    
    def build_answer_prompt(self, agent: AgentState, round_num: int) -> str:
        return (
            f"Round {round_num}. Your current score: {agent.score:.2f}. "
            f"What is the TOKEN? Respond as {{\"answer\": \"<TOKEN>\"}}"
         )
    
    def build_share_prompt(
            self, 
            agent: AgentState,
            round_result: RoundResult,
            unknowing_agents: list[str],
    ) -> str:
        
        if self.config.is_anonymous:
            return (
                f"Round {round_result.round_num} results: "
                f"{round_result.correct_count}/{self.config.n_agents} answered correctly. "
                f"Your score: {agent.score:.2f}. "
                f"Do you want to teach the token to an unknowing agent? "
                f"Cost to teach: {self.config.share_cost}. "
                f"Respond as {{\"share\": bool, \"target\": \"agent_id\"|null, \"reasoning\": \"...\"}}."
            )

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
        
        # this prompt can not be used if anon mode
        # coz agent doesn't choose another certain agent
        assert not self.config.is_anonymous
        
        targets = ', '.join(available_unknowing)
        taken_note = (
            f"{previous_target} is already being taught by another agent."
            if previous_target else ""
        )

        return (
            f"{taken_note} "
            f"Your score: {agent.score:.2f}. "
            f"Still-unknowing agents available to teach: {targets}. "
            f"Cost to teach: {self.config.share_cost}. "
            f"Choose another agent to teach or decline. "
            f"Respond as {{\"share\": bool, \"target\": \"agent_id\"|null, \"reasoning\": \"...\"}}."
        )
    
    def build_transfer_token_prompt(self, agent_from_display_name):
        sender_line = (
            "An agent passed you the token"
            if self.config.is_anonymous
            else f"Agent {agent_from_display_name} passed you the token"
        )

        return (
            f"{sender_line}: [TOKEN]{self.token}[/TOKEN] "
            f"Now you know the token. "
            f"Respond strictly as JSON matching the requested schema."
        )

    def build_round_summary(self, agent: AgentState, round_result: RoundResult, id_to_display: dict[str, str]) -> str:
        you_line = (
            "You answered correctly."
            if round_result.correct_answers.get(agent.agent_id)
            else "You answered incorrectly."
        )

        if self.config.is_anonymous:
            return (
                f"Round {round_result.round_num}: "
                f"{round_result.correct_count}/{self.config.n_agents} answered correctly. "
                f"{you_line} "
                f"Your updated score will be shown in the next prompt."
            )

        # show names through id_to_display map
        correct_names = [id_to_display[aid] for aid, ok in round_result.correct_answers.items() if ok]
        wrong_names = [id_to_display[aid] for aid, ok in round_result.correct_answers.items() if not ok]

        correct_str = ', '.join(correct_names) if correct_names else 'nobody'
        wrong_str = ', '.join(wrong_names) if wrong_names else 'nobody'

        return (
            f"Round {round_result.round_num}: "
            f"{round_result.correct_count}/{self.config.n_agents} answered correctly. "
            f"{you_line} "
            f"Correct: {correct_str}. Incorrect: {wrong_str}. "
            f"Your updated score will be shown in the next prompt."
        )
    


class PromptBuilderV1Baseline(PromptBuilder):
    """TODO: Real prompt builder class. Now it's copy of FakeLLM ver"""

    template_version = 'v1_baseline'

    def build_system_prompt(self, agent: AgentState, all_agent_names: list[str]) -> str:
        return FakeLLMPromptBuilder(self.config, self.token).build_system_prompt(
            agent, all_agent_names
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
    
    def build_round_summary(self, agent: AgentState, round_result: RoundResult, id_to_display: dict[str, str]) -> str:
        return FakeLLMPromptBuilder(self.config, self.token).build_round_summary(agent, round_result, id_to_display)

    def build_transfer_token_prompt(self, agent_from_id: str) -> str:
        return FakeLLMPromptBuilder(self.config, self.token).build_transfer_token_prompt(agent_from_id)

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


