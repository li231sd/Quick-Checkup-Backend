import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.image_converter import decode_base64_image
from models.abcde.inference import run_abcde_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def load_image_from_request(request):
    try:
        data = request.json
        if not data or 'image' not in data:
            return None, (jsonify({"error": "Missing image data"}), 400)
        image = decode_base64_image(data['image'])
        if image is None:
            return None, (jsonify({"error": "Invalid image format"}), 400)
        return image, None
    except Exception as e:
        return None, (jsonify({"status": "error", "message": str(e)}), 500)


def call_openrouter(prompt: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    res = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://medscan.app",
            "X-Title": "MedScan",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "max_tokens": 600,
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


def generate_analysis(abcde: dict) -> dict:
    prompt = f"""You are a dermatology assistant. Based on these ABCDE skin lesion scores (0-10 scale), generate a clinical analysis.

Scores:
- Asymmetry: {abcde['A']}/10
- Border: {abcde['B']}/10
- Color: {abcde['C']}/10
- Diameter: {abcde['D']}/10
- Evolution: {abcde['E']}/10
- Overall: {abcde['score']}/10

Respond ONLY with a JSON object, no markdown, no extra text:
{{
  "summary": "2-3 sentence plain-language overview of the lesion",
  "findings": "1-2 sentences describing what the scores indicate",
  "differentialDiagnosis": "1-2 sentences listing possible diagnoses based on scores",
  "nextSteps": "1-2 sentences recommending what the patient should do next"
}}"""

    import json
    text = call_openrouter(prompt)
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image, err = load_image_from_request(request)
        if err:
            return err

        abcde = run_abcde_model(image)
        abcde["score"] = round(sum(abcde.values()) / len(abcde), 2)

        analysis = generate_analysis(abcde)

        return jsonify({
            "status": "ok",
            "abcde": abcde,
            "summary": analysis["summary"],
            "findings": analysis["findings"],
            "differentialDiagnosis": analysis["differentialDiagnosis"],
            "nextSteps": analysis["nextSteps"],
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    