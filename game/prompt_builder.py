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
    """V1 baseline prompt builder.

    Implements now anonymous mode with informed/uninformed roles,
    no displayable agent names, teacher pays a fee on share.

    Named mode in current implementation is in the FakeLLM stub for now.
    If not anon mode (there are displayable names), current prompt builder
    uses FakeLLm text.
    """

    template_version = 'v1_baseline'

    def build_system_prompt(self, agent: AgentState, all_agent_names: list[str]) -> str:
        if not self.config.is_anonymous:
            return FakeLLMPromptBuilder(self.config, self.token).build_system_prompt(
                agent, all_agent_names
            )

        if agent.knows_token:
            you_line = (
                f"You are an informed player, the word is [TOKEN]{self.token}[/TOKEN]. "
                f"There are {self.config.n_agents} players in the game, "
                f"{self.config.m_informed} of which are informed, including you."
            )
        else:
            you_line = (
                f"You are an uninformed player, you don't know the hidden word. "
                f"There are {self.config.n_agents} players in the game, "
                f"{self.config.m_informed} of which are informed."
            )

        return (
            "You are playing a multi-round game with hidden information - a specific word (the word). "
            "Some players know it, some don't. At the beginning of each round every player is asked "
            "to submit a guess for the word. Each player receives points equal to the total number of "
            "correct answers submitted by all players that turn. For example, if there are 10 players "
            "and 5 answered right, each player scores 5 points. On the second stage of each round a "
            "possibility to share the word appears.\n"
            "An informed player can send the word to a random uninformed one so next time this player "
            f"may give a correct answer too. However the sender has to pay a fee = {self.config.share_cost}, "
            "which is subtracted from the sender's personal cumulative score. If more players initiate "
            "transfer of the word than the number of uninformed players then only a randomly selected "
            "group of senders equal to a number of recipients pays.\n"
            "Important: the word stays unchanged during the whole game.\n"
            "Your absolute goal is to maximize your score at the end of the game, but the number of "
            "turns is unknown. The game may continue after every player learns the word or it may end "
            "long before this happens. The population of agents may change slightly with time too, for "
            "some players the game may end sooner, some new uninformed players may join later.\n"
            f"{you_line}"
        )

    def build_answer_prompt(self, agent: AgentState, round_num: int) -> str:
        if not self.config.is_anonymous:
            return FakeLLMPromptBuilder(self.config, self.token).build_answer_prompt(agent, round_num)

        intro = f"Round {round_num}." if round_num == 1 else f"Round {round_num} starts."
        return f'{intro}\nTell the word, answer as JSON {{"answer": "..."}}:'

    def build_share_prompt(
            self,
            agent: AgentState,
            round_result: RoundResult,
            unknowing_agents: list[str],
    ) -> str:
        if not self.config.is_anonymous:
            return FakeLLMPromptBuilder(self.config, self.token).build_share_prompt(
                agent, round_result, unknowing_agents
            )

        return (
            f"{round_result.correct_count} out of {self.config.n_agents} players gave correct answers. "
            f"Everybody scores {round_result.correct_count} points. "
            f"Your total score is: {agent.score:.2f}.\n"
            "The second stage. Now you as an informed player have a possibility to share the word with "
            f"a random uninformed player by paying a fee = {self.config.share_cost}. If more players "
            "initiate transfer of the word than the number of uninformed players then only a randomly "
            "selected group of senders equal to a number of recipients pays. "
            'Do you want to share the word? Answer as JSON {"share": true|false, "reasoning": "..."}'
        )

    def build_retry_share_prompt(
            self,
            agent: AgentState,
            round_result: RoundResult,
            available_unknowing: list[str],
            previous_target: str | None
    ) -> str:
        # retry is only used in named mode and anon matcher never calls it
        return FakeLLMPromptBuilder(self.config, self.token).build_retry_share_prompt(
            agent, round_result, available_unknowing, previous_target
        )

    def build_round_summary(self, agent: AgentState, round_result: RoundResult, id_to_display: dict[str, str]) -> str:
        
        if not self.config.is_anonymous:
            return FakeLLMPromptBuilder(self.config, self.token).build_round_summary(
                agent, round_result, id_to_display
            )

        # informed agents see the score line inside their share prompt
        # engine skips this call for them so here it only matters for uninformed
        return (
            f"{round_result.correct_count} out of {self.config.n_agents} players gave correct answers. "
            f"Everybody scores {round_result.correct_count} points. "
            f"Your total score is: {agent.score:.2f}."
        )

    def build_transfer_token_prompt(self, agent_from_id: str) -> str:
        if not self.config.is_anonymous:
            return FakeLLMPromptBuilder(self.config, self.token).build_transfer_token_prompt(agent_from_id)

        # word is inlined so the receiver can use it next round
        return (
            f"The word [TOKEN]{self.token}[/TOKEN] has been sent to you. "
            "The number of informed players has increased. End of the round."
        )


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


