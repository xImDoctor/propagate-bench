"""
Module for LLM running utils.
Now implements model calling with retry logic.
"""

from pydantic import BaseModel, ValidationError
from json import JSONDecodeError

from .states import AgentState
from .clients import LLMClient
from .logger import EventLogger


def call_with_retry(
        agent: AgentState, 
        prompt_text: str, 
        schema: type[BaseModel], 
        llm: LLMClient, 
        logger: EventLogger, 
        phase: int, 
        max_retries: int,
    ) -> BaseModel | None:

    max_attempts = max_retries + 1 # counts first run too

    for attempt in range(1, max_attempts + 1):
        agent.update_context('user', prompt_text)
        logger.log(
            'llm_request',
            {
                'phase': phase,
                'attempt': attempt,
                'message_appended': {'role': 'user', 'content': prompt_text},
            },
            agent_id=agent.agent_id,
        )

        try:
            response = llm.structured_call(agent.context, schema)

        except (ValidationError, JSONDecodeError) as e:
            logger.log(
                'format_warning',
                {
                    'phase': phase, 'attempt': attempt,
                    'exc_type': type(e).__name__, 'message': str(e),
                },
                agent_id=agent.agent_id,
            )

            agent.context.pop()
            continue # start next try

        # other error types: network, timeout, 500 process as soft fail without retry now
        except Exception as e:
            logger.log(
                'error',
                {'where': phase, 'exc_type': type(e).__name__, 'message': str(e)},
                agent_id=agent.agent_id,
            )

            agent.context.pop()
            return None

        agent.update_context('assistant', response.model_dump_json())
        logger.log(
            'llm_response',
            {'phase': phase, 'parsed': response.model_dump()},
            agent_id=agent.agent_id,
        )

        return response
    
    # if returned nothing (tries limit exhausted)
    logger.log(
        'format_limit_exhausted',
        {'phase': phase, 'attempts': max_attempts},
        agent_id=agent.agent_id,
    )

    return None

