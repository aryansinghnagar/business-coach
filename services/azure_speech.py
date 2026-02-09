"""
Azure Speech Service module.

This module handles Azure Speech Service token generation and relay tokens
for avatar connections.
"""

import requests
from typing import Dict, Any, Optional
import config


class AzureSpeechService:
    """
    Service class for interacting with Azure Speech Service.
    
    Handles token generation for speech recognition and synthesis,
    as well as relay tokens for avatar WebRTC connections.
    """
    
    def __init__(self):
        """Initialize the Azure Speech Service."""
        self.speech_key = config.SPEECH_KEY
        self.speech_region = config.SPEECH_REGION
    
    def get_speech_token(self) -> Dict[str, str]:
        """
        Get an access token for Azure Speech Service.
        
        Returns:
            dict: Dictionary containing 'token' and 'region'
            
        Raises:
            requests.Timeout: If the request times out
            requests.RequestException: If the request fails
        """
        url = f"https://{self.speech_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        headers = {"Ocp-Apim-Subscription-Key": self.speech_key}
        
        try:
            resp = requests.post(url, headers=headers, timeout=5)
            resp.raise_for_status()
            token = resp.text.strip()
            
            if not token:
                raise ValueError("Empty token received from Azure Speech Service")
            
            return {
                "token": token,
                "region": self.speech_region
            }
        except requests.Timeout:
            raise requests.Timeout("Request to Azure Speech Service timed out")
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Failed to get speech token: {str(e)}"
            )
    
    def get_avatar_relay_token(self) -> Dict[str, Any]:
        """
        Get a WebRTC relay token for avatar connection.
        
        Returns:
            dict: Dictionary containing relay token information (Urls, Username, Password)
            
        Raises:
            requests.RequestException: If the request fails
        """
        url = (
            f"https://{self.speech_region}.tts.speech.microsoft.com/"
            f"cognitiveservices/avatar/relay/token/v1"
        )
        headers = {"Ocp-Apim-Subscription-Key": self.speech_key}
        
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Failed to get relay token: {str(e)}"
            )


# Lazy singleton: initialized on first use to avoid loading at import time
_speech_service: Optional[AzureSpeechService] = None


def get_speech_service() -> AzureSpeechService:
    """Return the Speech service instance, creating it on first call (lazy init)."""
    global _speech_service
    if _speech_service is None:
        _speech_service = AzureSpeechService()
    return _speech_service
