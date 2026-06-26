import random

from dataclasses import dataclass
from typing import Protocol

from .config import GameConfig
from .clients import LLMClient
from .states import GameState, RoundResult
from .prompt_builder import PromptBuilder, ShareResponse

from .llm_runner import call_with_retry

from .logger import EventLogger


@dataclass
class Transfer:
    from_id: str
    to_id: str
    cost_teacher: float
    cost_student: float = 0.0


class Matcher(Protocol):
    name: str

    def match(
            self,
            game_state: GameState,
            round_result: RoundResult,
            llm: LLMClient,
            prompts: PromptBuilder,
            logger: EventLogger,
            config: GameConfig,
            rng: random.Random,
    ) -> list[Transfer]: ...


class RandomChoiceMatcher:
    """Every knowing agent decides from the same state snapshot.
    Conflit resolving (2 knowings to 1 unknowing) through rng.choice"""

    name = 'random_choice'

    def match(
            self,
            game_state: GameState,
            round_result: RoundResult,
            llm: LLMClient,
            prompts: PromptBuilder,
            logger: EventLogger,
            config: GameConfig,
            rng: random.Random,
    ) -> list[Transfer]:
        
        unknowing_ids  = [agent.agent_id for agent in game_state.unknowing_agents()]
        if not unknowing_ids:
            return []
        
        decisions: dict[str, ShareResponse] = {}

        for agent in game_state.knowing_agents():
            
            prompt_text = prompts.build_share_prompt(agent, round_result, unknowing_ids)

            # prompt logging is inside of method
            response = call_with_retry(
                agent, prompt_text, ShareResponse,
                llm, logger, 'share', config.max_retries,
                extra_request_payload={'available': unknowing_ids},
            )

            if response is None:
                response = ShareResponse(share=False, target=None, reasoning='format_limit_exhausted')
            
            decisions[agent.agent_id] = response
            logger.log(
                'share_decision',
                response.model_dump(),
                agent_id=agent.agent_id,
            )

        target_to_candidates: dict[str, list[str]] = {}
        
        for sender_id, decision in decisions.items():
            if decision.share and decision.target in unknowing_ids:
                target_to_candidates.setdefault(decision.target, []).append(sender_id)
        
        transfers: list[Transfer] = []

        for target_id, candidates in target_to_candidates.items():
            winner_id = rng.choice(candidates)
            transfers.append(
                Transfer(
                    from_id=winner_id,
                    to_id=target_id,
                    cost_teacher=config.share_cost,
                    cost_student=0.0,
                )
            )

        return transfers


class FirstComeMatcher:
    """Every knowing agent decides from the same state snapshot.
    Conflit resolving (2 knowings to 1 unknowing) by selecting
    first knowing. Second one notes about it and can choose from
    other free unknowing agents or decline."""

    name = 'first_come'

    def match(
        self,
        game_state: GameState,
        round_result: RoundResult,
        llm: LLMClient,
        prompts: PromptBuilder,
        logger: EventLogger,
        config: GameConfig,
        rng: random.Random,
    ) -> list[Transfer]:

        unknowing_ids = [agent.agent_id for agent in game_state.unknowing_agents()]
        if not unknowing_ids:
            return []

        transfers: list[Transfer] = []
        taken: set[str] = set()

        pending = [agent.agent_id for agent in game_state.knowing_agents()]
        previous_target: dict[str, str] = {}
        is_first_round = True

        while pending:
            available_unknowings = [u for u in unknowing_ids if u not in taken]
            if not available_unknowings:
                break

            phase = 'share' if is_first_round else 'share_retry'

            decisions: dict[str, ShareResponse] = {}

            for teacher_id in pending:
                agent = game_state.get_agent(teacher_id)

                if is_first_round:
                    prompt_text = prompts.build_share_prompt(agent, round_result, available_unknowings)
                else:
                    prompt_text = prompts.build_retry_share_prompt(
                        agent, round_result, available_unknowings, previous_target.get(teacher_id)
                    )

                response = call_with_retry(
                    agent, prompt_text, ShareResponse,
                    llm, logger, phase, config.max_retries,
                    extra_request_payload={'available': available_unknowings},
                )
                if response is None:
                    response = ShareResponse(share=False, target=None, reasoning='format_limit_exhausted')

                logger.log(
                    'share_decision',
                    {**response.model_dump(), 'phase': phase},
                    agent_id=teacher_id,
                )

                decisions[teacher_id] = response

            # according in a survey order (first teacher of conflicted takes unknowing agent)
            next_pending: list[str] = []
            for teacher_id in pending:

                d = decisions[teacher_id]
                if not d.share or d.target not in available_unknowings:
                    continue # decline, invalide target
                if d.target in taken:
                    previous_target[teacher_id] = d.target
                    next_pending.append(teacher_id)
                    continue # already taken - retry

                taken.add(d.target)
                transfers.append(
                    Transfer(from_id=teacher_id, to_id=d.target,
                             cost_teacher=config.share_cost, cost_student=0.0)
                )

            pending = next_pending
            is_first_round = False

        return transfers

       

MATCHER_REGISTRY: dict[str, type] = {
    'random_choice': RandomChoiceMatcher,
    'first_come': FirstComeMatcher,
}


def create_matcher(config: GameConfig) -> Matcher:
    cls = MATCHER_REGISTRY.get(config.matcher)

    if cls is None:
        raise ValueError(
            f"Unknown matcher: {config.matcher!r}\n"
            f"Available matchers: {list(MATCHER_REGISTRY)}"
        )

    return cls()