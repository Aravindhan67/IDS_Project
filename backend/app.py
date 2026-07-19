from flask import Flask
from flask_cors import CORS

from routes.apk_routes import apk_bp

app = Flask(__name__)

CORS(app)

# Allow uploads up to 200 MB (default Flask limit is 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

app.register_blueprint(apk_bp)

@app.route("/")
def home():
    return {
        "status": "running",
        "project": "Android Malware Detection API"
    }

if __name__ == "__main__":
    app.run(debug=True)