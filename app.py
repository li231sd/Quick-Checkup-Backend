from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app) 

@app.route('/')
def health_check():
    return "AI Engine is Running!", 200

if __name__ == "__main__":
    # Railway sets the 'PORT' env var automatically
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
