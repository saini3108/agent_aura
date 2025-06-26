"""
Dynamic LLM Provider System
Supports Groq, OpenAI, Anthropic, and other providers
"""
import os
import json
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import tiktoken

@dataclass
class LLMResponse:
    content: str
    usage: Dict[str, int]
    model: str
    provider: str
    finish_reason: str = "stop"

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = None, 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class GroqProvider(LLMProvider):
    """Groq LLM Provider"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.default_model = "llama-3.1-70b-versatile"
        self.client = None
        
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("Groq package not installed. Install with: pip install groq")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = None, 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       **kwargs) -> LLMResponse:
        if not self.client:
            raise ValueError("Groq client not initialized. Check API key.")
        
        model = model or self.default_model
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=model,
                provider="groq",
                finish_reason=response.choices[0].finish_reason
            )
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
    
    def is_available(self) -> bool:
        return self.client is not None

class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.default_model = "gpt-4o"
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = None, 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       **kwargs) -> LLMResponse:
        if not self.client:
            raise ValueError("OpenAI client not initialized. Check API key.")
        
        model = model or self.default_model
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=model,
                provider="openai",
                finish_reason=response.choices[0].finish_reason
            )
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def is_available(self) -> bool:
        return self.client is not None

class AnthropicProvider(LLMProvider):
    """Anthropic LLM Provider"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.default_model = "claude-3-sonnet-20240229"
        self.client = None
        
        if self.api_key:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = None, 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       **kwargs) -> LLMResponse:
        if not self.client:
            raise ValueError("Anthropic client not initialized. Check API key.")
        
        model = model or self.default_model
        
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
            
            return LLMResponse(
                content=response.content[0].text,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                model=model,
                provider="anthropic",
                finish_reason=response.stop_reason
            )
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def is_available(self) -> bool:
        return self.client is not None

class LLMManager:
    """Manages multiple LLM providers with fallback support"""
    
    def __init__(self):
        self.providers = {}
        self.default_provider = None
        self.usage_stats = {}
        
    def add_provider(self, name: str, provider: LLMProvider, set_as_default: bool = False):
        """Add an LLM provider"""
        self.providers[name] = provider
        self.usage_stats[name] = {"requests": 0, "tokens": 0, "errors": 0}
        
        if set_as_default or not self.default_provider:
            self.default_provider = name
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [name for name, provider in self.providers.items() if provider.is_available()]
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       provider: str = None,
                       fallback: bool = True,
                       **kwargs) -> LLMResponse:
        """Generate chat completion with optional fallback"""
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found")
        
        try:
            response = self.providers[provider_name].chat_completion(messages, **kwargs)
            
            # Update usage stats
            self.usage_stats[provider_name]["requests"] += 1
            self.usage_stats[provider_name]["tokens"] += response.usage["total_tokens"]
            
            return response
            
        except Exception as e:
            self.usage_stats[provider_name]["errors"] += 1
            
            if fallback and len(self.get_available_providers()) > 1:
                # Try other available providers
                for fallback_provider in self.get_available_providers():
                    if fallback_provider != provider_name:
                        try:
                            response = self.providers[fallback_provider].chat_completion(messages, **kwargs)
                            self.usage_stats[fallback_provider]["requests"] += 1
                            self.usage_stats[fallback_provider]["tokens"] += response.usage["total_tokens"]
                            return response
                        except:
                            continue
            
            raise e
    
    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for all providers"""
        return self.usage_stats.copy()

def create_llm_manager() -> LLMManager:
    """Create and configure LLM manager with available providers"""
    manager = LLMManager()
    
    # Try to initialize providers based on available API keys
    if os.getenv("GROQ_API_KEY"):
        try:
            groq_provider = GroqProvider()
            manager.add_provider("groq", groq_provider, set_as_default=True)
        except Exception:
            pass
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai_provider = OpenAIProvider()
            manager.add_provider("openai", openai_provider)
        except Exception:
            pass
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            anthropic_provider = AnthropicProvider()
            manager.add_provider("anthropic", anthropic_provider)
        except Exception:
            pass
    
    return manager

# Global LLM manager instance
llm_manager = create_llm_manager()

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimation
        return len(text.split()) * 1.3

def truncate_text(text: str, max_tokens: int = 3000, model: str = "gpt-4") -> str:
    """Truncate text to fit within token limit"""
    current_tokens = count_tokens(text, model)
    
    if current_tokens <= max_tokens:
        return text
    
    # Rough truncation based on character ratio
    ratio = max_tokens / current_tokens
    truncated_length = int(len(text) * ratio * 0.9)  # 90% to be safe
    
    return text[:truncated_length] + "..."