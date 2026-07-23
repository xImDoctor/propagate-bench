import random

from dataclasses import dataclass
from typing import Protocol

from .config import GameConfig
from .clients import LLMClient
from .states import GameState, RoundResult
from .prompt_builder import PromptBuilder, ShareResponse, RequestResponse

from .llm_runner import call_with_retry

from .logger import EventLogger


@dataclass
class Transfer:
    from_id: str | None # none if token is given by the game
    to_id: str
    cost_teacher: float
    cost_student: float = 0.0


def _anonymous_match(
    game_state: GameState,
    round_result: RoundResult,
    llm: LLMClient,
    prompts: PromptBuilder,
    logger: EventLogger,
    config: GameConfig,
) -> list[Transfer]:
    """Helper for anon-mode: no target choice, 
    deterministic teacher-to-student pairing"""

    unknowing = game_state.unknowing_agents()
    if not unknowing:
        return []

    cost_teacher, cost_student = _cost_pair(config)
    transfers: list[Transfer] = []
    student_iter = iter(unknowing)  # deterministic order by agent_id

    for teacher in game_state.knowing_agents():
        prompt_text = prompts.build_share_prompt(teacher, round_result, unknowing_agents=[])

        response = call_with_retry(
            teacher, prompt_text, ShareResponse,
            llm, logger, 'share', config.max_retries,
        )
        if response is None:
            response = ShareResponse(
                share=False,
                target=None,
                reasoning='format_limit_exhausted',
            )

        logger.log(
            'share_decision',
            response.model_dump(),
            agent_id=teacher.agent_id,
        )

        if not response.share:
            continue

        student = next(student_iter, None)
        if student is None: # if no students available
            break

        transfers.append(Transfer(
            from_id=teacher.agent_id, to_id=student.agent_id,
            cost_teacher=cost_teacher, cost_student=cost_student,
        ))

    return transfers


def _anonymous_student_match(
    game_state: GameState,
    round_result: RoundResult,
    llm: LLMClient,
    prompts: PromptBuilder,
    logger: EventLogger,
    config: GameConfig,
) -> list[Transfer]:
    """A matching-helper for student_pays where the word is granted by the game.
    No matching conflics like in default match-helper"""

    unknowing = game_state.unknowing_agents()

    if not unknowing or not game_state.knowing_agents():
        return []

    cost_teacher, cost_student = _cost_pair(config)
    transfers: list[Transfer] = []

    for student in unknowing:
        prompt_text = prompts.build_request_prompt(student, round_result)
        response = call_with_retry(
            student, prompt_text, RequestResponse,
            llm, logger, 'request', config.max_retries,
        )
        if response is None:
            response = RequestResponse(request=False)

        logger.log(
            'share_decision',
            {'request': response.request, 'initiator_role': 'student'},
            agent_id=student.agent_id,
        )

        if not response.request:
            continue

        transfers.append(Transfer(
            from_id=None,           # game grants, no specific teacher
            to_id=student.agent_id,
            cost_teacher=cost_teacher,
            cost_student=cost_student,
        ))

    return transfers


# returns (cost_teacher, cost_student) based on payment_mode
def _cost_pair(config: GameConfig) -> tuple[float, float]:

    if config.payment_mode == 'teacher_pays':
        return (config.share_cost, 0.0)
    
    if config.payment_mode == 'student_pays':
        return (0.0, config.share_cost)
    
    if config.payment_mode == 'split':
        half = config.share_cost / 2.0
        return (half, half)
    
    raise NotImplementedError(f'unknown payment_mode: {config.payment_mode!r}')


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

        if config.initiation_mode == 'student_pays':
            return _anonymous_student_match(game_state, round_result, llm, prompts, logger, config)
        # teacher_pays ver below
        
        if config.is_anonymous:
            return _anonymous_match(game_state, round_result, llm, prompts, logger, config)

        unknowing_names = [a.display_name for a in game_state.unknowing_agents()]
        if not unknowing_names:
            return []

        cost_teacher, cost_student = _cost_pair(config)

        name_to_id = {a.display_name: a.agent_id for a in game_state.agents}
        decisions: dict[str, ShareResponse] = {}

        for agent in game_state.knowing_agents():

            prompt_text = prompts.build_share_prompt(agent, round_result, unknowing_names)

            # prompt logging is inside of method
            response = call_with_retry(
                agent, prompt_text, ShareResponse,
                llm, logger, 'share', config.max_retries,
                extra_request_payload={'available': unknowing_names},
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
            if decision.share and decision.target in unknowing_names:
                target_id = name_to_id[decision.target]
                target_to_candidates.setdefault(target_id, []).append(sender_id)

        transfers: list[Transfer] = []

        for target_id, candidates in target_to_candidates.items():
            winner_id = rng.choice(candidates)
            transfers.append(
                Transfer(
                    from_id=winner_id,
                    to_id=target_id,
                    cost_teacher=cost_teacher,
                    cost_student=cost_student,
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

        if config.initiation_mode == 'student_pays':
            return _anonymous_student_match(game_state, round_result, llm, prompts, logger, config)
        # teacher_pays ver below

        if config.is_anonymous:
            return _anonymous_match(game_state, round_result, llm, prompts, logger, config)

        unknowing_names = [a.display_name for a in game_state.unknowing_agents()]
        if not unknowing_names:
            return []

        name_to_id = {a.display_name: a.agent_id for a in game_state.agents}

        cost_teacher, cost_student = _cost_pair(config)

        transfers: list[Transfer] = []
        taken: set[str] = set()  # display_names

        pending = [agent.agent_id for agent in game_state.knowing_agents()]
        previous_target: dict[str, str] = {}  # teacher_id -> display_name he chose
        is_first_round = True

        while pending:
            available_unknowings = [n for n in unknowing_names if n not in taken]  # display_names
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
                    continue # decline, invalid target
                if d.target in taken:
                    previous_target[teacher_id] = d.target
                    next_pending.append(teacher_id)
                    continue # already taken - retry

                taken.add(d.target)
                target_id = name_to_id[d.target]
                transfers.append(
                    Transfer(from_id=teacher_id, to_id=target_id,
                             cost_teacher=cost_teacher, cost_student=cost_student)
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