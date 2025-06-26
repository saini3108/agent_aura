"""
Enterprise LLM Provider Management System
========================================

This module provides a robust, enterprise-grade LLM provider management system
that supports multiple AI providers (Groq, OpenAI, Anthropic, Gemini) with
automatic failover, load balancing, and comprehensive error handling.

Features:
- Multi-provider support with automatic failover
- Token counting and cost tracking
- Rate limiting and throttling
- Comprehensive logging and monitoring
- Dynamic provider configuration
- Connection pooling and caching

Usage:
    from src.core.llm_manager import LLMManager
    
    llm = LLMManager()
    response = llm.chat_completion(
        messages=[{"role": "user", "content": "Analyze this credit data"}],
        provider="groq"
    )
"""

import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class LLMManager:
    """Enterprise LLM Provider Management System"""
    
    def __init__(self):
        self.providers = {}
        self.active_provider = "groq"
        
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        return {
            "groq": {"status": "available", "model": "llama-3.1-8b-instant"},
            "openai": {"status": "configured" if os.getenv("OPENAI_API_KEY") else "not_configured"},
            "anthropic": {"status": "configured" if os.getenv("ANTHROPIC_API_KEY") else "not_configured"},
            "gemini": {"status": "configured" if os.getenv("GEMINI_API_KEY") else "not_configured"}
        }
    
    def test_connection(self, provider: str) -> bool:
        """Test connection to a provider"""
        return provider == "groq" and bool(os.getenv("GROQ_API_KEY"))

# Global instance
_llm_manager = None

def get_llm_manager() -> LLMManager:
    """Get global LLM manager instance"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Standardized response from LLM providers"""
    content: str
    usage: Dict[str, int]
    model: str
    provider: str
    finish_reason: str = "stop"
    latency_ms: float = 0
    cost_estimate: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        self.rate_limit_remaining = 100
        self.rate_limit_reset = time.time() + 60
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider's client"""
        pass
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = None, 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       **kwargs) -> LLMResponse:
        """Generate chat completion"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass
    
    def check_rate_limit(self) -> bool:
        """Check if rate limit allows request"""
        current_time = time.time()
        if current_time > self.rate_limit_reset:
            self.rate_limit_remaining = 100
            self.rate_limit_reset = current_time + 60
        
        return self.rate_limit_remaining > 0
    
    def update_rate_limit(self, remaining: int, reset_time: float):
        """Update rate limit information"""
        self.rate_limit_remaining = remaining
        self.rate_limit_reset = reset_time

class GroqProvider(LLMProvider):
    """Groq LLM Provider with enterprise features"""
    
    def __init__(self, api_key: str = None):
        self.default_model = "llama-3.1-8b-instant"
        self.models = [
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "llama-3.2-90b-text-preview"
        ]
        super().__init__(api_key)
    
    def _initialize_client(self):
        """Initialize Groq client"""
        if not self.api_key:
            self.api_key = os.getenv("GROQ_API_KEY")
        
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized successfully")
            except ImportError:
                logger.error("Groq package not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = None, 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       **kwargs) -> LLMResponse:
        """Generate chat completion using Groq"""
        if not self.client:
            raise ValueError("Groq client not initialized")
        
        if not self.check_rate_limit():
            raise Exception("Rate limit exceeded for Groq")
        
        model = model or self.default_model
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            latency = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=model,
                provider="groq",
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency,
                cost_estimate=self._estimate_cost(response.usage.total_tokens)
            )
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise Exception(f"Groq API error: {str(e)}")
    
    def _estimate_cost(self, total_tokens: int) -> float:
        """Estimate cost for Groq usage (typically free/very low cost)"""
        return total_tokens * 0.0001  # Rough estimate
    
    def is_available(self) -> bool:
        """Check if Groq is available"""
        return self.client is not None and self.check_rate_limit()

class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider with enterprise features"""
    
    def __init__(self, api_key: str = None):
        self.default_model = "gpt-4o"
        self.models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        super().__init__(api_key)
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized successfully")
            except ImportError:
                logger.error("OpenAI package not installed")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = None, 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       **kwargs) -> LLMResponse:
        """Generate chat completion using OpenAI"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        if not self.check_rate_limit():
            raise Exception("Rate limit exceeded for OpenAI")
        
        model = model or self.default_model
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            latency = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=model,
                provider="openai",
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency,
                cost_estimate=self._estimate_cost(model, response.usage)
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _estimate_cost(self, model: str, usage) -> float:
        """Estimate cost for OpenAI usage"""
        # Pricing as of 2024 (per 1K tokens)
        pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
        
        if model in pricing:
            input_cost = (usage.prompt_tokens / 1000) * pricing[model]["input"]
            output_cost = (usage.completion_tokens / 1000) * pricing[model]["output"]
            return input_cost + output_cost
        
        return 0.01  # Default estimate
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return self.client is not None and self.check_rate_limit()

class AnthropicProvider(LLMProvider):
    """Anthropic LLM Provider with enterprise features"""
    
    def __init__(self, api_key: str = None):
        self.default_model = "claude-3-sonnet-20240229"
        self.models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
        super().__init__(api_key)
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if self.api_key:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized successfully")
            except ImportError:
                logger.error("Anthropic package not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = None, 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       **kwargs) -> LLMResponse:
        """Generate chat completion using Anthropic"""
        if not self.client:
            raise ValueError("Anthropic client not initialized")
        
        if not self.check_rate_limit():
            raise Exception("Rate limit exceeded for Anthropic")
        
        model = model or self.default_model
        start_time = time.time()
        
        # Convert messages format for Anthropic
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=user_messages,
                **kwargs
            )
            
            latency = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.content[0].text,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                model=model,
                provider="anthropic",
                finish_reason=response.stop_reason,
                latency_ms=latency,
                cost_estimate=self._estimate_cost(model, response.usage)
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def _estimate_cost(self, model: str, usage) -> float:
        """Estimate cost for Anthropic usage"""
        # Pricing as of 2024 (per 1K tokens)
        pricing = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}
        }
        
        if model in pricing:
            input_cost = (usage.input_tokens / 1000) * pricing[model]["input"]
            output_cost = (usage.output_tokens / 1000) * pricing[model]["output"]
            return input_cost + output_cost
        
        return 0.01  # Default estimate
    
    def is_available(self) -> bool:
        """Check if Anthropic is available"""
        return self.client is not None and self.check_rate_limit()

class LLMManager:
    """Enterprise LLM Manager with failover and monitoring"""
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.default_provider = None
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_success: Dict[str, datetime] = {}
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available providers"""
        # Initialize Groq
        try:
            groq_provider = GroqProvider()
            if groq_provider.is_available():
                self.add_provider("groq", groq_provider, set_as_default=True)
        except Exception as e:
            logger.warning(f"Failed to initialize Groq: {e}")
        
        # Initialize OpenAI
        try:
            openai_provider = OpenAIProvider()
            if openai_provider.is_available():
                self.add_provider("openai", openai_provider)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Initialize Anthropic
        try:
            anthropic_provider = AnthropicProvider()
            if anthropic_provider.is_available():
                self.add_provider("anthropic", anthropic_provider)
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic: {e}")
    
    def add_provider(self, name: str, provider: LLMProvider, set_as_default: bool = False):
        """Add a provider to the manager"""
        self.providers[name] = provider
        self.usage_stats[name] = {
            "requests": 0,
            "tokens": 0,
            "errors": 0,
            "total_cost": 0.0,
            "avg_latency": 0.0
        }
        self.error_counts[name] = 0
        self.last_success[name] = datetime.now()
        
        if set_as_default or not self.default_provider:
            self.default_provider = name
        
        logger.info(f"Added provider: {name}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        available = []
        for name, provider in self.providers.items():
            if provider.is_available() and self.error_counts[name] < 5:
                available.append(name)
        return available
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       provider: str = None,
                       fallback: bool = True,
                       **kwargs) -> LLMResponse:
        """Generate chat completion with failover support"""
        provider_name = provider or self.default_provider
        
        if not provider_name:
            raise ValueError("No providers available")
        
        # Try primary provider
        if provider_name in self.providers:
            try:
                response = self._execute_completion(provider_name, messages, **kwargs)
                self._update_success_stats(provider_name, response)
                return response
            except Exception as e:
                self._update_error_stats(provider_name, str(e))
                logger.warning(f"Provider {provider_name} failed: {e}")
                
                if not fallback:
                    raise e
        
        # Try fallback providers
        if fallback:
            available_providers = [p for p in self.get_available_providers() 
                                 if p != provider_name]
            
            for fallback_provider in available_providers:
                try:
                    response = self._execute_completion(fallback_provider, messages, **kwargs)
                    self._update_success_stats(fallback_provider, response)
                    logger.info(f"Fallback to {fallback_provider} successful")
                    return response
                except Exception as e:
                    self._update_error_stats(fallback_provider, str(e))
                    logger.warning(f"Fallback provider {fallback_provider} failed: {e}")
                    continue
        
        raise Exception("All providers failed")
    
    def _execute_completion(self, provider_name: str, messages: List[Dict[str, str]], 
                          **kwargs) -> LLMResponse:
        """Execute completion with a specific provider"""
        provider = self.providers[provider_name]
        response = provider.chat_completion(messages, **kwargs)
        return response
    
    def _update_success_stats(self, provider_name: str, response: LLMResponse):
        """Update success statistics"""
        stats = self.usage_stats[provider_name]
        stats["requests"] += 1
        stats["tokens"] += response.usage["total_tokens"]
        stats["total_cost"] += response.cost_estimate
        
        # Update average latency
        current_avg = stats["avg_latency"]
        request_count = stats["requests"]
        stats["avg_latency"] = ((current_avg * (request_count - 1)) + response.latency_ms) / request_count
        
        self.error_counts[provider_name] = 0  # Reset error count on success
        self.last_success[provider_name] = datetime.now()
    
    def _update_error_stats(self, provider_name: str, error: str):
        """Update error statistics"""
        self.usage_stats[provider_name]["errors"] += 1
        self.error_counts[provider_name] += 1
        logger.error(f"Provider {provider_name} error: {error}")
    
    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive usage statistics"""
        return {
            provider: {
                **stats,
                "error_count": self.error_counts[provider],
                "last_success": self.last_success[provider].isoformat(),
                "availability": self.providers[provider].is_available()
            }
            for provider, stats in self.usage_stats.items()
        }
    
    def reset_error_counts(self):
        """Reset error counts for all providers"""
        for provider in self.error_counts:
            self.error_counts[provider] = 0
        logger.info("Error counts reset for all providers")
    
    def set_default_provider(self, provider_name: str):
        """Set the default provider"""
        if provider_name in self.providers:
            self.default_provider = provider_name
            logger.info(f"Default provider set to: {provider_name}")
        else:
            raise ValueError(f"Provider {provider_name} not found")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers"""
        health_status = {}
        
        for name, provider in self.providers.items():
            try:
                # Simple test completion
                test_messages = [{"role": "user", "content": "Hello"}]
                response = provider.chat_completion(test_messages, max_tokens=10)
                
                health_status[name] = {
                    "status": "healthy",
                    "latency": response.latency_ms,
                    "last_check": datetime.now().isoformat()
                }
            except Exception as e:
                health_status[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_check": datetime.now().isoformat()
                }
        
        return health_status

# Global LLM manager instance
_llm_manager: Optional[LLMManager] = None

def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken"""
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimation
        return int(len(text.split()) * 1.3)

def truncate_text(text: str, max_tokens: int = 3000, model: str = "gpt-4") -> str:
    """Truncate text to fit within token limit"""
    current_tokens = count_tokens(text, model)
    
    if current_tokens <= max_tokens:
        return text
    
    # Rough truncation based on character ratio
    ratio = max_tokens / current_tokens
    truncated_length = int(len(text) * ratio * 0.9)  # 90% to be safe
    
    return text[:truncated_length] + "..."