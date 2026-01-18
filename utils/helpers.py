"""
Helper utility functions.

This module contains reusable utility functions used throughout the application.
"""

from typing import Dict, Any
import config


def build_config_response() -> Dict[str, Any]:
    """
    Build a complete configuration response dictionary.
    
    This function aggregates all configuration settings into a single
    dictionary for the /config/all endpoint.
    
    Returns:
        dict: Complete configuration dictionary with all service settings
    """
    cog_search_config = config.get_cognitive_search_config()
    
    return {
        "speech": {
            "region": config.SPEECH_REGION,
            "privateEndpointEnabled": config.SPEECH_PRIVATE_ENDPOINT_ENABLED
        },
        "openai": {
            "endpoint": config.AZURE_OPENAI_ENDPOINT,
            "deploymentName": config.DEPLOYMENT_NAME,
            "apiVersion": "2023-06-01-preview"
        },
        "cognitiveSearch": {
            "enabled": cog_search_config.get("enabled", False),
            "endpoint": cog_search_config.get("endpoint"),
            "apiKey": cog_search_config.get("apiKey"),
            "indexName": cog_search_config.get("indexName")
        },
        "sttTts": {
            "sttLocales": config.STT_LOCALES,
            "ttsVoice": config.TTS_VOICE,
            "customVoiceEndpointId": config.CUSTOM_VOICE_ENDPOINT_ID,
            "continuousConversation": config.CONTINUOUS_CONVERSATION
        },
        "avatar": {
            "character": config.AVATAR_CHARACTER,
            "style": config.AVATAR_STYLE,
            "photoAvatar": config.PHOTO_AVATAR,
            "customized": config.CUSTOMIZED_AVATAR,
            "useBuiltInVoice": config.USE_BUILT_IN_VOICE,
            "autoReconnect": config.AUTO_RECONNECT_AVATAR,
            "useLocalVideoForIdle": config.USE_LOCAL_VIDEO_FOR_IDLE,
            "showSubtitles": config.SHOW_SUBTITLES
        },
        "systemPrompt": config.SYSTEM_PROMPT,
        "faceDetection": config.get_face_detection_config(),
        "azureFaceApi": config.get_azure_face_api_config()
    }
