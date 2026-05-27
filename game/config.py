from typing import Literal
from pydantic import BaseModel, ConfigDict, model_validator


class GameConfig(BaseModel):

    model_config = ConfigDict(frozen=True, extra='forbid')

    n_agents: int
    m_informed: int
    share_cost: float
    token: str | None = None # by us or randomized if None
    seed: int = 42
    max_rounds: int = 1

    agent_names: list[str] | None = None

    # LLM
    model: str
    api_type: Literal['ollama', 'together', 'fake']
    # TODO: talk abt LLM params
    # top_p: float = 1.0
    # temperature: float = 0.7
    request_timeout: float = 60.0 # sec
    max_retries: int = 1

    # prompts and matching
    template_version: str = 'prompt_v1'
    # random choice matcher resolves conflicts of 2 knowings -> 1 unknowing through rng.choice
    matcher: Literal['random_choice', 'mutual_consent'] = 'random_choice'

    # experiment modes
    initiation_mode: Literal['teacher_only', 'student_only', 'both'] = 'teacher_only'
    payment_mode: Literal['teacher_pays', 'student_pays', 'split'] = 'teacher_pays'
    token_transfer_mode: Literal['direct', 'dialog'] = 'direct'


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
        
        if self.agent_names is not None:
            if len(self.agent_names) != self.n_agents:
                raise ValueError('agent_names length must be equal no agent number')
            if len(set(self.agent_names)) != len(self.agent_names):
                raise ValueError('agent_names must be unique')

        # stubs for modes that not implemented yet
        if self.matcher != 'random_choice':
            raise NotImplementedError(f"matcher={self.matcher!r} not implemented yet")
        if self.initiation_mode != 'teacher_only':
            raise NotImplementedError(f"initiation_mode={self.initiation_mode!r} not implemented yet")
        if self.payment_mode != 'teacher_pays':
            raise NotImplementedError(f"payment_mode={self.payment_mode!r} not implemented yet")
        if self.token_transfer_mode != 'direct':
            raise NotImplementedError(f"transfer_mode={self.token_transfer_mode!r} not implemented yet")

        return self
        