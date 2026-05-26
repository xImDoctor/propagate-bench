import subprocess
from pathlib import Path
import random

from .config import GameConfig

from .llm_client import LLMClient
from .states import AgentState, GameState, RoundResult

from .prompt_builder import AnswerResponse, PromptBuilder, create_prompt_builder
from .agent_matching import Matcher, create_matcher

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
        self.logger.log(
            'game_init',
            {
                'config': self.config.model_dump(),
                'token': self.token,
                'informed_agent_ids': [agent.agent_id for agent in self.game_state.knowing_agents()],
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
        
        while not self.game_state.game_over:
            self.game_state.round += 1
            self.logger.set_round(self.game_state.round)
            self.logger.log('round_start', {})

            round_result = self._run_answer_phase()
            self._run_scoring_phase(round_result)
            self._run_communcation_phase(round_result)

            self.logger.log(
                'round_summary',
                {
                    'answers': round_result.answers,
                    'correct': round_result.correct,
                    'scores_after': round_result.scores_after,
                }
            )

            self.game_state.check_stop(self.config.max_rounds)

        stop_reason = 'all_know' if all(agent.knows_token for agent in self.game_state.agents) else 'max_rounds'
        self.logger.log(
            'game_over',
            {
            'reason': stop_reason,
            'n_rounds': self.game_state.round,
            'final_scores': {agent.agent_id: agent.score for agent in self.game_state.agents},
            },
        )


    def _run_answer_phase(self): ...

    def _run_scoring_phase(self, round_result): ...

    def _run_communcation_phase(self, round_result): ...