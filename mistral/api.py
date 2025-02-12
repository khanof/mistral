from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Change the Mistral vLLM API to its internal container port
VLLM_API_URL = "http://localhost:8001/generate"

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 200)

    # Send request to vLLM API
    response = requests.post(VLLM_API_URL, json={"prompt": prompt, "max_tokens": max_tokens})

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": "Failed to generate response"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Now runs on port 5000
