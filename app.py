import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
# Assuming you create an inference.py with these functions
# from inference import run_abcde_model, run_pattern_model 

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- MOCK MODELS FOR HACKATHON DEMO ---
# Replace these with your actual model logic calls
def run_abcde_model(image_data):
    # Logic for Asymmetry, Border, Color, etc.
    return {"risk": "high", "score": 0.82, "details": "Irregular Border detected"}

def run_pattern_model(image_data):
    # Logic for Deep Learning Pattern Recognition
    return {"risk": "low", "score": 0.35}

@app.route('/', methods=['GET'])
def health():
    return "AI Consensus Engine is Online", 200

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # 1. Run both models (Dual Model approach)
        result_a = run_abcde_model(data['image'])
        result_b = run_pattern_model(data['image'])

        # 2. Consensus Logic: "If either flags high risk, we alert"
        is_malignant = (result_a['risk'] == "high" or result_b['risk'] == "high")
        
        # 3. Secure Database Handoff (The 'Cloud' part of your diagram)
        # Here you would typically save to Mongo/Postgres
        
        return jsonify({
            "status": "success",
            "prediction": "Malignant" if is_malignant else "Benign",
            "consensus_reached": True,
            "analysis": {
                "abcde_check": result_a,
                "pattern_recognition": result_b
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Binding to 0.0.0.0 is required for Railway
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
