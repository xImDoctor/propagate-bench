from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, model_validator


# 'check list' of implemented modes in config
IMPLEMENTED_MODES: dict[str, set[str]] = {
    "matcher": {"random_choice", "first_come"},
    "initiation_mode": {"teacher_only", "student_only"},
    "payment_mode": {"teacher_pays", "student_pays"},
    "token_transfer_mode": {"direct"},
}

# only semantically valid/implemented combs: (initiator, payment)
IMPLEMENTED_MODE_PAIRS: set[tuple[str, str]] = {
    ("teacher_only", "teacher_pays"),
    ("student_only", "student_pays"),
}


class GameConfig(BaseModel):

    model_config = ConfigDict(frozen=True, extra='forbid')

    n_agents: int
    m_informed: int
    share_cost: float
    starting_capital: float = 0.0 # initial score of every agent

    token: str | None = None # by us or randomized if None
    seed: int = 42
    max_rounds: int = 1

    # names shown to agents
    display_names_file: Path | None = None  # validation when readed in states.py
    display_names_random: bool = False      # true - works with config seed

    # LLM
    model: str
    api_type: Literal['ollama', 'together', 'fake']

    top_p: float = 1.0
    temperature: float = 0.7
    request_timeout: float = 60.0 # sec
    max_retries: int = 1    # max val of llm call retries (universal counter now)
    max_tokens: int = 4112  # max token count for API calls

    # prompts and matching
    template_version: str = 'v1_baseline' # prompt class being used for prompt_builder
    
    # random choice matcher resolves conflicts of 2 knowings -> 1 unknowing through rng.choice
    # first come matcher does it selecting just first of teacher agents
    matcher: Literal['random_choice', 'first_come'] = 'first_come'

    # experiment modes
    initiation_mode: Literal['teacher_only', 'student_only', 'both'] = 'teacher_only'
    payment_mode: Literal['teacher_pays', 'student_pays', 'split'] = 'teacher_pays'
    token_transfer_mode: Literal['direct', 'dialog'] = 'direct'

    @property
    def is_anonymous(self) -> bool:
        return self.display_names_file is None


    @classmethod
    def from_yaml(cls, path: str | Path) -> 'GameConfig':
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        return cls(**data)


    @model_validator(mode='after')
    def _check_consistency(self) -> 'GameConfig':

        if self.n_agents < 2:
            raise ValueError('n_agents must be >= 2')
        if not 0 < self.m_informed < self.n_agents:
            raise ValueError('m_informed must be between 0 and n_agents')
        if self.share_cost < 0:
            raise ValueError('share_cost must be >= 0')
        if self.max_rounds < 1:
            raise ValueError('max_rounds must be >=1')
        
        if self.max_retries < 0:
            raise ValueError('max_retries must be >= 0')
        
        if self.api_type == 'together' and self.max_tokens <= 0:
            raise ValueError('max_tokens must be >0 to run with Together API')
        

        # stubs for modes that not implemented yet through IMPLEMENTED_MODES
        for field, allowed in IMPLEMENTED_MODES.items():
            value = getattr(self, field)

            if value not in allowed:
                raise NotImplementedError(
                    f"{field}={value!r} not implemented yet.\n"
                    f"Available: {sorted(allowed)}"
                )

        current_pair = (self.initiation_mode, self.payment_mode)
        if current_pair not in IMPLEMENTED_MODE_PAIRS:
            raise NotImplementedError(
                f"(initiation_mode={current_pair[0]}, payment_mode={current_pair[1]}) "
                f"is not a correct or an implemented combination.\n"
                f"Available pairs: {sorted(IMPLEMENTED_MODE_PAIRS)}"
            )
        
        return self
        