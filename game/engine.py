import subprocess
from pathlib import Path
import random

from .config import GameConfig

from .clients import LLMClient
from .states import GameState, RoundResult

from .prompt_builder import AnswerResponse, PromptBuilder, create_prompt_builder
from .agent_matching import Matcher, create_matcher

from .llm_runner import call_with_retry, FormatLimitExhausted

from .logger import EventLogger


# try to extract last commit SHA
def _git_commit_hash() -> str | None:
    
    try:
        output = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                timeout=10, # sec
        )
        return output.stdout.strip() if output.returncode == 0 else None
    
    except Exception:
        return None
    

class GameEngine:
    def __init__(self, config: GameConfig, llm: LLMClient, logger: EventLogger, prompts: PromptBuilder | None = None, matcher: Matcher | None = None):
        self.config = config
        self.llm = llm
        self.logger = logger
        self.rng = random.Random(config.seed)

        # token from config or just int from [1000, 9999]
        self.token = config.token or str(self.rng.randint(1000, 9999))

        self.game_state = GameState.initialize(config, self.rng)
        self.prompts = prompts or create_prompt_builder(config, self.token)
        self.matcher = matcher or create_matcher(config)

        self._init_agents()


    def _init_agents(self) -> None:
        all_ids = [agent.agent_id for agent in self.game_state.agents]
        all_names = [agent.display_name for agent in self.game_state.agents]

        self.logger.log(
            'game_init',
            {
                'config': self.config.model_dump(),
                'token': self.token,
                'informed_agent_ids': [agent.agent_id for agent in self.game_state.knowing_agents()],
                'display_names': {agent.agent_id: agent.display_name for agent in self.game_state.agents},
                'last_git_commit_hash': _git_commit_hash(),
            },
        )

        for agent in self.game_state.agents:
    
            system_prompt = self.prompts.build_system_prompt(agent, all_ids)

            agent.update_context('system', system_prompt)
            self.logger.log(
                'system_prompt_set',
                {'text': system_prompt},
                agent_id=agent.agent_id,
            )

    def run(self) -> None:

        stop_reason: str
        stuck_info: dict | None = None
        
        try: 
            while not self.game_state.game_over:
                self.game_state.round += 1
                self.logger.set_round(self.game_state.round)
                self.logger.log('round_start', {})

                round_result = self._run_answer_phase()
                self._run_scoring_phase(round_result)
                self._run_communication_phase(round_result)

                self.logger.log(
                    'round_summary',
                    {
                        'answers': round_result.answers,
                        'correct': round_result.correct_answers,
                        'scores_after': round_result.scores_after,
                    }
                )

                self.game_state.check_stop(self.config.max_rounds)

            stop_reason = 'all_know' if all(agent.knows_token for agent in self.game_state.agents) else 'max_rounds'
        
        # stops game if one of agents stuck on schema format
        except FormatLimitExhausted as e:
            stop_reason = 'format_limit_exhausted'
            stuck_info = {
                'stuck_agent_id': e.agent_id,
                'stuck_phase': e.phase,
                'stuck_attempts': e.attempts,
            }

        payload = {
            'reason': stop_reason,
            'n_rounds': self.game_state.round,
            'final_scores': {agent.agent_id: agent.score for agent in self.game_state.agents},
        }

        if stuck_info:
            payload.update(stuck_info) # swap to stuck_info if agent stuck

        self.logger.log(
            'game_over',
            payload,
        )


    # TODO: try to parallel answer phase (parallel requests to cloud API, not ollama)
    def _run_answer_phase(self) -> RoundResult:
        answers: dict[str, str] = {}
        correct: dict[str, bool] = {}

        for agent in self.game_state.agents:
            prompt_text = self.prompts.build_answer_prompt(agent, self.game_state.round)

            # prompt logging is inside of method, it provides schema validation and retry logic
            response = call_with_retry(
                agent, prompt_text, AnswerResponse, self.llm, self.logger, 
                'answer', self.config.max_retries,
            )
            answer_str = response.answer if response else ''
            is_correct = answer_str.strip() == self.token
            answers[agent.agent_id] = answer_str
            correct[agent.agent_id] = is_correct

            self.logger.log(
                'answer',
                {'answer': answer_str, 'is_correct': is_correct},
                agent_id=agent.agent_id,
            )

            if agent.knows_token and not is_correct:
                self.logger.log(
                    'informed_agent_wrong',
                    {'expected_token': self.token, 'actual_answer': answer_str},
                    agent_id=agent.agent_id,
                )

        return RoundResult(
            round_num=self.game_state.round,
            answers=answers,
            correct_answers=correct,
            scores_after={},
            correct_count=sum(correct.values()),
        )
        

    def _run_scoring_phase(self, round_result : RoundResult) -> None:

        self.game_state.distribute_score(round_result.correct_count)
        
        round_result.scores_after = {agent.agent_id: agent.score for agent in self.game_state.agents}
        
        gain = round_result.correct_count / self.config.n_agents
        self.logger.log(
            'score_update',
            {
                'correct_count': round_result.correct_count,
                'gain': gain,
                'scores_after': round_result.scores_after,
            },
        )

        id_to_display = {agent.agent_id: agent.display_name for agent in self.game_state.agents}

        for agent in self.game_state.agents:
            round_summary = self.prompts.build_round_summary(agent, round_result, id_to_display)
            agent.update_context('user', round_summary)

            self.logger.log(
                'summary_to_context',
                {'phase': 'round_summary', 'content': round_summary},
                agent_id=agent.agent_id
            )


    def _run_communication_phase(self, _round_result : RoundResult) -> None:
        
        transfers = self.matcher.match(
            game_state=self.game_state,
            round_result=_round_result,
            llm=self.llm,
            prompts=self.prompts,
            logger=self.logger,
            config=self.config,
            rng=self.rng,
        )

        for t in transfers:
            self.game_state.apply_transfer(t.from_id, t.to_id, t.cost_teacher, t.cost_student)
            self.logger.log(
                'token_received',
                {
                    'from': t.from_id,
                    'to': t.to_id,
                    'cost_teacher': t.cost_teacher,
                    'cost_student': t.cost_student,
                },
            )

            # prompt_text_transfer_token = self.prompts.build_transfer_token_prompt(t.to_id)
            sender = self.game_state.get_agent(t.from_id)
            prompt_text_transfer_token = self.prompts.build_transfer_token_prompt(sender.display_name)
            
            receiver = self.game_state.get_agent(t.to_id)
            receiver.update_context('user', prompt_text_transfer_token)
            self.logger.log(
                'summary_to_context',
                {
                    'phase': 'token_received', 'content': prompt_text_transfer_token
                },
                agent_id=t.to_id,
            )

