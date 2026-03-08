import os
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
        
        # Add overall score as average of A B C D E
        abcde["score"] = round(sum(abcde.values()) / len(abcde), 2)

        return jsonify({
            "status": "ok",
            "abcde": abcde
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
