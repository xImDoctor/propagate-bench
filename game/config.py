from typing import Literal
from pydantic import BaseModel, ConfigDict, model_validator


class GameConfig(BaseModel):

    model_config = ConfigDict(frozen=True, extra='forbid')

    n_agents: int
    m_informed: int
    share_cost: float

    # by us or randomized if None
    token: str | None = None

    show_names: bool = True
    seed: int = 42

    max_rounds: int = 1

    # LLM
    # TODO: talk abt model params and top_p/temperature for model determination
    # model: str
    # api_type: Literal["ollama", "together", "fake"]
    # top_p: float = 1.0
    # temperature: float = 0.7
    # request_timeout: float = 60.0 # sec
    # max_retries: int = 1

    # prompt naming/versioning
    template_version: str = "prompt_v1"


    @model_validator(mode="after")
    def _check_consistency(self) -> "GameConfig":

        if self.n_agents < 2:
            raise ValueError("n_agents must be >= 2")
        if not 0 < self.m_informed < self.n_agents:
            raise ValueError("m_informed must be between 0 and n_agents")
        if self.share_cost < 0:
            raise ValueError("share_cost must be >= 0")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >=1")
        
        return self
        