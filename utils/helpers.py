"""
Helper utility functions.

This module contains reusable utility functions used throughout the application.
"""

from typing import Dict, Any
import config
from utils.face_detection_preference import get_face_detection_method


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
        "foundry": {
            "endpoint": config.AZURE_FOUNDRY_ENDPOINT,
            "deploymentName": config.FOUNDRY_DEPLOYMENT_NAME,
            "apiVersion": "2023-06-01-preview"
        },
        "openai": {  # backward compatibility alias
            "endpoint": config.AZURE_FOUNDRY_ENDPOINT,
            "deploymentName": config.FOUNDRY_DEPLOYMENT_NAME,
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
        "faceDetection": {
            **config.get_face_detection_config(),
            "method": get_face_detection_method(),
            "lightweightMode": config.LIGHTWEIGHT_MODE,
        },
        "signifierWeights": {
            "url": config.SIGNIFIER_WEIGHTS_URL,
            "path": config.SIGNIFIER_WEIGHTS_PATH,
        },
        "azureFaceApi": config.get_azure_face_api_config(),
        "acoustic": {
            "acousticAnalysisEnabled": config.ACOUSTIC_ANALYSIS_ENABLED,
            "acousticContextMaxAgeSec": config.ACOUSTIC_CONTEXT_MAX_AGE_SEC,
        },
    }
