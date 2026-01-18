"""
Azure OpenAI service module.

This module handles all interactions with Azure OpenAI, including
chat completions and streaming responses with support for On Your Data.
"""

import requests
from typing import List, Dict, Any, Generator, Optional
from openai import AzureOpenAI
import config


class AzureOpenAIService:
    """
    Service class for interacting with Azure OpenAI.
    
    Handles both standard and streaming chat completions, with optional
    support for Azure Cognitive Search (On Your Data).
    """
    
    def __init__(self):
        """Initialize the Azure OpenAI client."""
        self.client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
        self.endpoint = config.AZURE_OPENAI_ENDPOINT
        self.deployment_name = config.DEPLOYMENT_NAME
        self.api_key = config.AZURE_OPENAI_KEY
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Get a non-streaming chat completion from Azure OpenAI.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt to prepend to messages
            
        Returns:
            str: The assistant's response content
            
        Raises:
            Exception: If the API call fails
        """
        # Add system prompt if provided and not already in messages
        if system_prompt:
            if not any(msg.get("role") == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
        )
        return response.choices[0].message.content
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        enable_oyd: bool = False,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Get a streaming chat completion from Azure OpenAI.
        
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
        
        # Build the base URL
        if enable_oyd and config.is_cognitive_search_enabled():
            url = (
                f"{self.endpoint}openai/deployments/{self.deployment_name}/"
                f"extensions/chat/completions?api-version=2023-06-01-preview"
            )
        else:
            url = (
                f"{self.endpoint}openai/deployments/{self.deployment_name}/"
                f"chat/completions?api-version=2023-06-01-preview"
            )
        
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


# Global service instance
openai_service = AzureOpenAIService()
