import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import AzureOpenAI

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Azure OpenAI Configuration
AZURE_OPENAI_KEY = "C5udZNFPjx2mIwtcfgcbu9llK45zslpah0iFIKU7ryvcMwGoC1MAJQQJ99CAACHYHv6XJ3w3AAABACOGObhF"
AZURE_OPENAI_ENDPOINT = "https://openai-agent137.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-4o"

# Azure Speech Configuration
SPEECH_KEY = "9imU7HnZLctrGSq8z0qBWRypGyBp8OpOAH6WAFfYkPo72qoFHoBFJQQJ99CAACHYHv6XJ3w3AAAYACOGjQfN"
SPEECH_REGION = "eastus2"

# Azure Cognitive Search Configuration (optional - for On Your Data)
AZURE_COG_SEARCH_ENDPOINT = ""
AZURE_COG_SEARCH_API_KEY = ""
AZURE_COG_SEARCH_INDEX_NAME = ""

# STT / TTS Configuration
STT_LOCALES = "en-US,de-DE,es-ES,fr-FR,it-IT,ja-JP,ko-KR,zh-CN"
TTS_VOICE = "en-US-AvaMultilingualNeural"
CUSTOM_VOICE_ENDPOINT_ID = ""
CONTINUOUS_CONVERSATION = False

# Avatar Configuration
AVATAR_CHARACTER = "lisa"
AVATAR_STYLE = "casual-sitting"
PHOTO_AVATAR = False
CUSTOMIZED_AVATAR = False
USE_BUILT_IN_VOICE = False
AUTO_RECONNECT_AVATAR = False
USE_LOCAL_VIDEO_FOR_IDLE = False
SHOW_SUBTITLES = False

# System Prompt
SYSTEM_PROMPT = "You are an AI assistant that helps people find information."

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
)


@app.route("/")
def index():
    try:
        return send_from_directory(".", "index.html")
    except FileNotFoundError:
        return jsonify({"error": "index.html not found"}), 404


@app.route("/chat.html")
def chat_page():
    try:
        return send_from_directory(".", "chat.html")
    except FileNotFoundError:
        return jsonify({"error": "chat.html not found"}), 404


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/chat", methods=["POST"])
def chat():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json(silent=True) or {}
    user_input = data.get("message", "").strip()
    
    if not user_input:
        return jsonify({"error": "Missing message"}), 400

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a business coach. Keep responses concise."},
                {"role": "user", "content": user_input},
            ],
        )
        return jsonify({"response": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": "Failed to get AI response", "details": str(e)}), 500


@app.route("/speech/token", methods=["GET"])
def get_speech_token():
    url = f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    headers = {"Ocp-Apim-Subscription-Key": SPEECH_KEY}
    
    try:
        resp = requests.post(url, headers=headers, timeout=5)
        resp.raise_for_status()
        token = resp.text.strip()
        if not token:
            return jsonify({"error": "Empty token received"}), 502
        return jsonify({"token": token, "region": SPEECH_REGION})
    except requests.Timeout:
        return jsonify({"error": "Request timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": "Failed to get speech token", "details": str(e)}), 502


@app.route("/config/speech", methods=["GET"])
def get_speech_config():
    """Return speech configuration (region only, key stays on backend)"""
    return jsonify({
        "region": SPEECH_REGION,
        "privateEndpointEnabled": False
    })


@app.route("/config/openai", methods=["GET"])
def get_openai_config():
    """Return OpenAI configuration (endpoint and deployment only, key stays on backend)"""
    return jsonify({
        "endpoint": AZURE_OPENAI_ENDPOINT,
        "deploymentName": DEPLOYMENT_NAME,
        "apiVersion": "2023-06-01-preview"
    })


@app.route("/config/cognitive-search", methods=["GET"])
def get_cognitive_search_config():
    """Return Cognitive Search configuration if available"""
    if AZURE_COG_SEARCH_ENDPOINT and AZURE_COG_SEARCH_API_KEY and AZURE_COG_SEARCH_INDEX_NAME:
        return jsonify({
            "endpoint": AZURE_COG_SEARCH_ENDPOINT,
            "apiKey": AZURE_COG_SEARCH_API_KEY,
            "indexName": AZURE_COG_SEARCH_INDEX_NAME,
            "enabled": True
        })
    return jsonify({"enabled": False})


@app.route("/config/stt-tts", methods=["GET"])
def get_stt_tts_config():
    """Return STT/TTS configuration"""
    return jsonify({
        "sttLocales": STT_LOCALES,
        "ttsVoice": TTS_VOICE,
        "customVoiceEndpointId": CUSTOM_VOICE_ENDPOINT_ID,
        "continuousConversation": CONTINUOUS_CONVERSATION
    })


@app.route("/config/avatar", methods=["GET"])
def get_avatar_config():
    """Return Avatar configuration"""
    return jsonify({
        "character": AVATAR_CHARACTER,
        "style": AVATAR_STYLE,
        "photoAvatar": PHOTO_AVATAR,
        "customized": CUSTOMIZED_AVATAR,
        "useBuiltInVoice": USE_BUILT_IN_VOICE,
        "autoReconnect": AUTO_RECONNECT_AVATAR,
        "useLocalVideoForIdle": USE_LOCAL_VIDEO_FOR_IDLE,
        "showSubtitles": SHOW_SUBTITLES
    })


@app.route("/config/system-prompt", methods=["GET"])
def get_system_prompt():
    """Return system prompt"""
    return jsonify({
        "prompt": SYSTEM_PROMPT
    })


@app.route("/config/all", methods=["GET"])
def get_all_config():
    """Return all configuration in one endpoint"""
    cog_search_enabled = bool(AZURE_COG_SEARCH_ENDPOINT and AZURE_COG_SEARCH_API_KEY and AZURE_COG_SEARCH_INDEX_NAME)
    return jsonify({
        "speech": {
            "region": SPEECH_REGION,
            "privateEndpointEnabled": False
        },
        "openai": {
            "endpoint": AZURE_OPENAI_ENDPOINT,
            "deploymentName": DEPLOYMENT_NAME,
            "apiVersion": "2023-06-01-preview"
        },
        "cognitiveSearch": {
            "enabled": cog_search_enabled,
            "endpoint": AZURE_COG_SEARCH_ENDPOINT if cog_search_enabled else None,
            "apiKey": AZURE_COG_SEARCH_API_KEY if cog_search_enabled else None,
            "indexName": AZURE_COG_SEARCH_INDEX_NAME if cog_search_enabled else None
        },
        "sttTts": {
            "sttLocales": STT_LOCALES,
            "ttsVoice": TTS_VOICE,
            "customVoiceEndpointId": CUSTOM_VOICE_ENDPOINT_ID,
            "continuousConversation": CONTINUOUS_CONVERSATION
        },
        "avatar": {
            "character": AVATAR_CHARACTER,
            "style": AVATAR_STYLE,
            "photoAvatar": PHOTO_AVATAR,
            "customized": CUSTOMIZED_AVATAR,
            "useBuiltInVoice": USE_BUILT_IN_VOICE,
            "autoReconnect": AUTO_RECONNECT_AVATAR,
            "useLocalVideoForIdle": USE_LOCAL_VIDEO_FOR_IDLE,
            "showSubtitles": SHOW_SUBTITLES
        },
        "systemPrompt": SYSTEM_PROMPT
    })


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming chat endpoint with support for conversation history and On Your Data"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    enable_oyd = data.get("enableOyd", False)
    system_prompt = data.get("systemPrompt", "You are an AI assistant that helps people find information.")
    
    if not messages:
        return jsonify({"error": "Missing messages"}), 400
    
    # Add system message if not present
    if not any(msg.get("role") == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})
    
    try:
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version=2023-06-01-preview"
        headers = {
            "api-key": AZURE_OPENAI_KEY,
            "Content-Type": "application/json"
        }
        
        body = {
            "messages": messages,
            "stream": True
        }
        
        # Add data sources for On Your Data if enabled
        if enable_oyd and AZURE_COG_SEARCH_ENDPOINT:
            url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{DEPLOYMENT_NAME}/extensions/chat/completions?api-version=2023-06-01-preview"
            body["dataSources"] = [{
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": AZURE_COG_SEARCH_ENDPOINT,
                    "key": AZURE_COG_SEARCH_API_KEY,
                    "indexName": AZURE_COG_SEARCH_INDEX_NAME,
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
        
        response = requests.post(url, headers=headers, json=body, stream=True)
        response.raise_for_status()
        
        def generate():
            try:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        if line.strip() == '[DONE]':
                            yield 'data: [DONE]\n\n'
                        elif line.startswith('data:'):
                            yield line + '\n\n'
                        else:
                            yield 'data: ' + line + '\n\n'
                yield 'data: [DONE]\n\n'
            except Exception as e:
                yield f'data: {{"error": "Streaming error: {str(e)}"}}\n\n'
        
        return app.response_class(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        })
    except Exception as e:
        return jsonify({"error": "Failed to get AI response", "details": str(e)}), 500


@app.route("/avatar/relay-token", methods=["GET"])
def get_avatar_relay_token():
    """Get WebRTC relay token for avatar connection"""
    url = f"https://{SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1"
    headers = {"Ocp-Apim-Subscription-Key": SPEECH_KEY}
    
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"error": "Failed to get relay token", "details": str(e)}), 502


if __name__ == "__main__":
    app.run(port=5000, debug=True)
