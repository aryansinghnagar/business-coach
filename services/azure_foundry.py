"""
Azure AI Foundry service module.

This module handles all interactions with Azure AI Foundry (upgraded from Azure OpenAI),
including chat completions and streaming responses with support for On Your Data.
"""

import requests
from typing import List, Dict, Any, Generator, Optional
from openai import AzureOpenAI
import config


class AzureFoundryService:
    """
    Service class for interacting with Azure AI Foundry.
    
    Handles both standard and streaming chat completions, with optional
    support for Azure Cognitive Search (On Your Data).
    """
    
    def __init__(self):
        """Initialize the Azure AI Foundry client."""
        self.client = AzureOpenAI(
            azure_endpoint=config.AZURE_FOUNDRY_ENDPOINT,
            api_key=config.AZURE_FOUNDRY_KEY,
            api_version=config.AZURE_FOUNDRY_API_VERSION,
        )
        self.endpoint = config.AZURE_FOUNDRY_ENDPOINT
        self.deployment_name = config.FOUNDRY_DEPLOYMENT_NAME
        self.api_key = config.AZURE_FOUNDRY_KEY
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        api_version_override: Optional[str] = None,
    ) -> str:
        """
        Get a non-streaming chat completion from Azure AI Foundry.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt to prepend to messages
            max_tokens: Optional max response length (e.g. for concise insights)
            temperature: Optional sampling temperature (0-1); higher = more varied output
            api_version_override: Optional API version (e.g. 2024-08-01-preview) for 404 fallback
            
        Returns:
            str: The assistant's response content
            
        Raises:
            Exception: If the API call fails
        """
        client = self.client
        if api_version_override:
            client = AzureOpenAI(
                azure_endpoint=config.AZURE_FOUNDRY_ENDPOINT,
                api_key=config.AZURE_FOUNDRY_KEY,
                api_version=api_version_override,
            )
        # Add system prompt if provided and not already in messages
        if system_prompt:
            if not any(msg.get("role") == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})
        kwargs = {"model": self.deployment_name, "messages": messages}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        enable_oyd: bool = False,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Get a streaming chat completion from Azure AI Foundry.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            enable_oyd: Whether to enable On Your Data (Azure Cognitive Search)
            system_prompt: Optional system prompt (defaults to config value)
            
        Yields:
            str: Server-Sent Events formatted data chunks
            
        Raises:
            Exception: If the API call fails
        """
        # Use provided system prompt or default from config
        if system_prompt is None:
            system_prompt = config.SYSTEM_PROMPT
        
        # Ensure system message is present
        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Build the base URL (endpoint has no trailing slash; add / before path)
        base = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment_name}/"
        if enable_oyd and config.is_cognitive_search_enabled():
            url = base + "extensions/chat/completions?api-version=2023-06-01-preview"
        else:
            url = base + "chat/completions?api-version=2023-06-01-preview"
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        body = {
            "messages": messages,
            "stream": True
        }
        
        # Add data sources for On Your Data if enabled
        if enable_oyd and config.is_cognitive_search_enabled():
            body["dataSources"] = [{
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": config.AZURE_COG_SEARCH_ENDPOINT,
                    "key": config.AZURE_COG_SEARCH_API_KEY,
                    "indexName": config.AZURE_COG_SEARCH_INDEX_NAME,
                    "semanticConfiguration": "",
                    "queryType": "simple",
                    "fieldsMapping": {
                        "contentFieldsSeparator": "\n",
                        "contentFields": ["content"],
                        "filepathField": None,
                        "titleField": "title",
                        "urlField": None
                    },
                    "inScope": True,
                    "roleInformation": system_prompt
                }
            }]
        
        # Make streaming request
        response = requests.post(url, headers=headers, json=body, stream=True)
        response.raise_for_status()
        
        # Stream the response
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.strip() == '[DONE]':
                    yield 'data: [DONE]\n\n'
                elif line.startswith('data:'):
                    yield line + '\n\n'
                else:
                    yield 'data: ' + line + '\n\n'
        
        # Final done signal
        yield 'data: [DONE]\n\n'


# Lazy singleton: initialized on first use to avoid loading Azure SDK at import time
_foundry_service: Optional[AzureFoundryService] = None


def get_foundry_service() -> AzureFoundryService:
    """Return the Azure AI Foundry service instance, creating it on first call (lazy init)."""
    global _foundry_service
    if _foundry_service is None:
        _foundry_service = AzureFoundryService()
    return _foundry_service


# Backward compatibility alias (deprecated; use get_foundry_service)
def get_openai_service() -> AzureFoundryService:
    """Deprecated: use get_foundry_service(). Kept for backward compatibility."""
    return get_foundry_service()
