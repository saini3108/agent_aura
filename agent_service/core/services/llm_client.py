import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
import os

from agent_service.core.config import settings

logger = logging.getLogger(__name__)

class ModelProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    usage: Dict[str, Any]
    model: str
    provider: str
    metadata: Dict[str, Any]

class BaseLLMClient(ABC):
    """Base class for LLM clients"""

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.provider = self.get_provider()

    @abstractmethod
    def get_provider(self) -> str:
        """Get provider name"""
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from prompt"""
        pass

    @abstractmethod
    async def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> LLMResponse:
        """Generate structured output"""
        pass

class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""

    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        super().__init__(api_key, model_name)

        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
            self.openai = openai
        except ImportError:
            logger.error("OpenAI package not installed")
            raise

    def get_provider(self) -> str:
        return ModelProvider.OPENAI.value

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                usage=response.usage.__dict__,
                model=response.model,
                provider=self.provider,
                metadata={"response_id": response.id}
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> LLMResponse:
        """Generate structured output using OpenAI API"""
        try:
            # Add schema to prompt
            structured_prompt = f"""
            {prompt}

            Please respond with a JSON object that matches this schema:
            {json.dumps(schema, indent=2)}

            Your response must be valid JSON.
            """

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": structured_prompt}],
                **kwargs
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                usage=response.usage.__dict__,
                model=response.model,
                provider=self.provider,
                metadata={"response_id": response.id, "schema": schema}
            )

        except Exception as e:
            logger.error(f"OpenAI structured generation error: {e}")
            raise

class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client"""

    def __init__(self, api_key: str, model_name: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key, model_name)

        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            logger.error("Anthropic package not installed")
            raise

    def get_provider(self) -> str:
        return ModelProvider.ANTHROPIC.value

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Anthropic API"""
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1000),
                messages=[{"role": "user", "content": prompt}],
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )

            return LLMResponse(
                content=response.content[0].text,
                usage=response.usage.__dict__,
                model=response.model,
                provider=self.provider,
                metadata={"response_id": response.id}
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> LLMResponse:
        """Generate structured output using Anthropic API"""
        try:
            structured_prompt = f"""
            {prompt}

            Please respond with a JSON object that matches this schema:
            {json.dumps(schema, indent=2)}

            Your response must be valid JSON.
            """

            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1000),
                messages=[{"role": "user", "content": structured_prompt}],
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )

            return LLMResponse(
                content=response.content[0].text,
                usage=response.usage.__dict__,
                model=response.model,
                provider=self.provider,
                metadata={"response_id": response.id, "schema": schema}
            )

        except Exception as e:
            logger.error(f"Anthropic structured generation error: {e}")
            raise

class GoogleClient(BaseLLMClient):
    """Google Gemini API client"""

    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        super().__init__(api_key, model_name)

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name)
            self.genai = genai
        except ImportError:
            logger.error("Google AI package not installed")
            raise

    def get_provider(self) -> str:
        return ModelProvider.GOOGLE.value

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Google Gemini API"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.client.generate_content, prompt
            )

            return LLMResponse(
                content=response.text,
                usage={"prompt_tokens": 0, "completion_tokens": 0},  # Gemini doesn't provide usage
                model=self.model_name,
                provider=self.provider,
                metadata={"response_id": "gemini_response"}
            )

        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise

    async def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> LLMResponse:
        """Generate structured output using Google Gemini API"""
        try:
            structured_prompt = f"""
            {prompt}

            Please respond with a JSON object that matches this schema:
            {json.dumps(schema, indent=2)}

            Your response must be valid JSON.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None, self.client.generate_content, structured_prompt
            )

            return LLMResponse(
                content=response.text,
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                model=self.model_name,
                provider=self.provider,
                metadata={"response_id": "gemini_response", "schema": schema}
            )

        except Exception as e:
            logger.error(f"Google structured generation error: {e}")
            raise

class GroqClient(BaseLLMClient):
    """Groq API client"""

    def __init__(self, api_key: str, model_name: str = "groq-default"):
        super().__init__(api_key, model_name)
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except ImportError:
            logger.error("Groq not installed")
            raise

    def get_provider(self) -> str:
        return ModelProvider.GROQ.value

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.generate(prompt=prompt, model=self.model_name, **kwargs)
            )
            return LLMResponse(
                content=response.text,
                usage=getattr(response, "usage", {}),
                model=response.model,
                provider=self.provider,
                metadata={"response_id": getattr(response, "id", None)}
            )
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    async def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> LLMResponse:
        structured_prompt = (
            f"{prompt}\n\nPlease respond with a JSON object matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\nYour response must be valid JSON."
        )
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.generate(prompt=structured_prompt, model=self.model_name, **kwargs)
            )
            return LLMResponse(
                content=response.text,
                usage=getattr(response, "usage", {}),
                model=response.model,
                provider=self.provider,
                metadata={"response_id": getattr(response, "id", None), "schema": schema}
            )
        except Exception as e:
            logger.error(f"Groq structured generation error: {e}")
            raise

class LLMClientManager:
    """Manager for multiple LLM clients"""

    def __init__(self):
        self.clients: Dict[str, BaseLLMClient] = {}
        self.initialize_clients()

    def initialize_clients(self) -> None:
        """Initialize available LLM clients"""

        # OpenAI
        if settings.OPENAI_API_KEY:
            try:
                self.clients["openai"] = OpenAIClient(
                    api_key=settings.OPENAI_API_KEY,
                    model_name=settings.DEFAULT_MODEL_NAME if settings.DEFAULT_MODEL_PROVIDER == "openai" else "gpt-4"
                )
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

        # Anthropic
        if settings.ANTHROPIC_API_KEY:
            try:
                self.clients["anthropic"] = AnthropicClient(
                    api_key=settings.ANTHROPIC_API_KEY,
                    model_name=settings.DEFAULT_MODEL_NAME if settings.DEFAULT_MODEL_PROVIDER == "anthropic" else "claude-3-sonnet-20240229"
                )
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")

        # Google
        if settings.GOOGLE_API_KEY:
            try:
                self.clients["google"] = GoogleClient(
                    api_key=settings.GOOGLE_API_KEY,
                    model_name=settings.DEFAULT_MODEL_NAME if settings.DEFAULT_MODEL_PROVIDER == "google" else "gemini-pro"
                )
                logger.info("Google client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google client: {e}")

        # Groq
        if settings.GROQ_API_KEY:
            try:
                self.clients["groq"] = GroqClient(
                    api_key=settings.GROQ_API_KEY,
                    model_name=settings.DEFAULT_MODEL_NAME if settings.DEFAULT_MODEL_PROVIDER == "groq" else "groq-default"
                )
                logger.info("Groq client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")


    def get_client(self, provider: Optional[str] = None) -> BaseLLMClient:
        """Get LLM client by provider"""
        provider = provider or settings.DEFAULT_MODEL_PROVIDER

        if provider not in self.clients:
            raise ValueError(f"Provider {provider} not available. Available: {list(self.clients.keys())}")

        return self.clients[provider]

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.clients.keys())

    async def generate(self, prompt: str, provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate text using specified provider"""
        client = self.get_client(provider)
        return await client.generate(prompt, **kwargs)

    async def generate_structured(self, prompt: str, schema: Dict[str, Any], provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate structured output using specified provider"""
        client = self.get_client(provider)
        return await client.generate_structured(prompt, schema, **kwargs)

# Global LLM client manager
llm_manager = LLMClientManager()
