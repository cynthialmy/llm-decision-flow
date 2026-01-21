"""Base agent class with Azure OpenAI client and common utilities."""
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Optional, Dict, Any
import json
import logging
from openai import AzureOpenAI
from pydantic import BaseModel, ValidationError
from src.config import get_azure_openai_client, settings, get_foundry_project_client, get_foundry_agent_name

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class BaseAgent(ABC):
    """Base class for all agents with Azure OpenAI integration."""

    def __init__(self):
        """Initialize agent with Azure OpenAI client."""
        self.client: AzureOpenAI = get_azure_openai_client()
        self.deployment_name: str = settings.azure_openai_deployment_name

        # Check if we should use Foundry agent
        self.foundry_project_client = get_foundry_project_client()
        self.foundry_agent_name = get_foundry_agent_name()
        self.use_foundry_agent = self.foundry_project_client is not None and self.foundry_agent_name is not None

        if self.use_foundry_agent:
            logger.info(f"Using Foundry agent: {self.foundry_agent_name}")

    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Call Azure OpenAI API or Foundry agent with retry logic.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            response_format: Response format (e.g., {"type": "json_object"})

        Returns:
            Response text from LLM
        """
        # Use Foundry agent if available
        if self.use_foundry_agent:
            return self._call_foundry_agent(prompt, system_prompt)

        # Standard Azure OpenAI approach
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI API: {e}")
            raise

    def _call_foundry_agent(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Call Foundry agent using agent reference.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional, may be handled by agent)

        Returns:
            Response text from agent
        """
        try:
            # Get the agent
            agent = self.foundry_project_client.agents.get(agent_name=self.foundry_agent_name)

            # Prepare input messages
            input_messages = []
            if system_prompt:
                input_messages.append({"role": "system", "content": system_prompt})
            input_messages.append({"role": "user", "content": prompt})

            # Call agent via OpenAI client
            response = self.client.responses.create(
                input=input_messages,
                extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
            )

            return response.output_text or ""
        except Exception as e:
            logger.error(f"Error calling Foundry agent: {e}")
            raise

    def _parse_structured_output(
        self,
        response_text: str,
        output_model: Type[T],
        retry_on_error: bool = True
    ) -> T:
        """
        Parse LLM response into structured Pydantic model.

        Args:
            response_text: Raw response text from LLM
            output_model: Pydantic model class to parse into
            retry_on_error: Whether to retry parsing if JSON parsing fails

        Returns:
            Parsed Pydantic model instance
        """
        try:
            # Try to extract JSON from response if it's wrapped in markdown
            json_text = response_text.strip()
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()

            # Parse JSON
            data = json.loads(json_text)

            # Validate and create Pydantic model
            return output_model(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Error parsing structured output: {e}")
            logger.error(f"Response text: {response_text}")

            if retry_on_error:
                # Try to fix common JSON issues and retry once
                try:
                    # Remove trailing commas, fix quotes, etc.
                    json_text = json_text.replace(",\n}", "\n}").replace(",\n]", "\n]")
                    data = json.loads(json_text)
                    return output_model(**data)
                except Exception:
                    pass

            raise ValueError(f"Failed to parse LLM response into {output_model.__name__}: {e}")

    def _call_llm_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        output_model: Type[T] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> T:
        """
        Call LLM and parse response into structured Pydantic model.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            output_model: Pydantic model class to parse into
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response

        Returns:
            Parsed Pydantic model instance
        """
        if output_model is None:
            raise ValueError("output_model must be provided")

        # Request JSON format for structured output
        response_format = {"type": "json_object"}

        # Enhance system prompt to request JSON output
        enhanced_system = system_prompt or ""
        if enhanced_system:
            enhanced_system += "\n\nIMPORTANT: Respond with valid JSON only, matching the expected schema."
        else:
            enhanced_system = "Respond with valid JSON only, matching the expected schema."

        response_text = self._call_llm(
            prompt=prompt,
            system_prompt=enhanced_system,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )

        return self._parse_structured_output(response_text, output_model)

    @abstractmethod
    def process(self, *args, **kwargs):
        """Process input and return agent output. Must be implemented by subclasses."""
        pass
