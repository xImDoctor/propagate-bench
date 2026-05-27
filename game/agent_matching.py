import random

from dataclasses import dataclass
from typing import Protocol

from .config import GameConfig
from .llm_client import LLMClient
from .states import GameState, RoundResult
from .prompt_builder import PromptBuilder, ShareResponse

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
            
            agent.update_context('user', prompt_text)
            logger.log(
                'llm_request',
                {'phase': 'share', 'message_appended': {'role': 'user', 'content': prompt_text}},
                agent_id=agent.agent_id
            )

            try:
                response = llm.structured_call(agent.context, ShareResponse)
                assert isinstance(response, ShareResponse) # check schema
            
            except Exception as e:
                logger.log(
                    'error',
                    {'where': 'share_phase', 'exc_type': type(e).__name__, 'message': str(e)},
                    agent_id=agent.agent_id,
                )
                # silent fallback without game breaking
                response = ShareResponse(share=False, target=None, reasoning='error')

            agent.update_context('assistant', response.model_dump_json())
            logger.log(
                'llm_response',
                {"phase": "share", "parsed": response.model_dump()},
                agent_id=agent.agent_id,
            )
            
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
    

MATCHER_REGISTRY: dict[str, type] = {
    'random_choice': RandomChoiceMatcher,
}


def create_matcher(config: GameConfig) -> Matcher:
    cls = MATCHER_REGISTRY.get(config.matcher)

    if cls is None:
        raise ValueError(
            f"Unknown matcher: {config.matcher!r}\n"
            f"Available matchers: {list(MATCHER_REGISTRY)}"
        )

    return cls()