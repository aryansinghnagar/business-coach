"""
Azure Speech Service module.

Handles Azure Speech Service token generation for speech recognition (STT).
Client speech is converted to text and sent to the backend for transcript storage
and speech cue analysis (Azure Foundry). Avatar/TTS and relay tokens have been
removed; insights are text-only.
"""

import requests
from typing import Dict, Optional
import config


class AzureSpeechService:
    """
    Service class for Azure Speech Service.
    Provides tokens for speech recognition (STT) only.
    """

    def __init__(self):
        """Initialize the Azure Speech Service."""
        self.speech_key = config.SPEECH_KEY
        self.speech_region = config.SPEECH_REGION

    def get_speech_token(self) -> Dict[str, str]:
        """
        Get an access token for Azure Speech Service (STT).

        Returns:
            dict: Dictionary containing 'token' and 'region'

        Raises:
            ValueError: If SPEECH_KEY or SPEECH_REGION is not configured
            requests.Timeout: If the request times out
            requests.RequestException: If the request fails
        """
        if not (self.speech_key and self.speech_key.strip()):
            raise ValueError(
                "Speech service is not configured. Set SPEECH_KEY in your environment (e.g. in .env)."
            )
        if not (self.speech_region and self.speech_region.strip()):
            raise ValueError(
                "Speech region is not set. Set SPEECH_REGION in your environment (e.g. centralindia)."
            )
        url = f"https://{self.speech_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        headers = {"Ocp-Apim-Subscription-Key": self.speech_key}

        try:
            resp = requests.post(url, headers=headers, timeout=5)
            if resp.status_code == 401:
                raise ValueError(
                    "Azure returned 401 Permission Denied. Check that SPEECH_KEY is a valid key "
                    "from your Azure Speech resource and SPEECH_REGION matches that resource's region "
                    "(e.g. centralindia, eastus). Get keys in Azure Portal: create or open a Speech resource, "
                    "then Keys and Endpoint."
                )
            resp.raise_for_status()
            token = resp.text.strip()

            if not token:
                raise ValueError("Empty token received from Azure Speech Service")

            return {
                "token": token,
                "region": self.speech_region
            }
        except ValueError:
            raise
        except requests.Timeout:
            raise requests.Timeout("Request to Azure Speech Service timed out")
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Failed to get speech token: {str(e)}"
            )


# Lazy singleton: initialized on first use to avoid loading at import time
_speech_service: Optional[AzureSpeechService] = None


def get_speech_service() -> AzureSpeechService:
    """Return the Speech service instance, creating it on first call (lazy init)."""
    global _speech_service
    if _speech_service is None:
        _speech_service = AzureSpeechService()
    return _speech_service
