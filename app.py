from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = r"E:\WINTER SEMESTER 24-45\DIGITAL IMAGE PROCESSING\archive\saved_models\sports_detection.h5"  # Ensure this path is correct
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
class_names = [
    "air hockey", "ampute football", "archery", "arm wrestling", "axe throwing",
    "balance beam", "barell racing", "baseball", "basketball", "baton twirling",
    "bike polo", "billiards", "bmx", "bobsled", "bowling", "boxing", "bull riding",
    "bungee jumping", "canoe slamon", "cheerleading", "chuckwagon racing", "cricket",
    "croquet", "curling", "disc golf", "fencing", "field hockey", "figure skating men",
    "figure skating pairs", "figure skating women", "fly fishing", "football",
    "formula 1 racing", "frisbee", "gaga", "giant slalom", "golf", "hammer throw",
    "hang gliding", "harness racing", "high jump", "hockey", "horse jumping",
    "horse racing", "horseshoe pitching", "hurdles", "hydroplane racing",
    "ice climbing", "ice yachting", "jai alai", "javelin", "jousting", "judo",
    "lacrosse", "log rolling", "luge", "motorcycle racing", "mushing", "nascar racing",
    "olympic wrestling", "parallel bar", "pole climbing", "pole dancing", "pole vault",
    "polo", "pommel horse", "rings", "rock climbing", "roller derby",
    "rollerblade racing", "rowing", "rugby", "sailboat racing", "shot put",
    "shuffleboard", "sidecar racing", "ski jumping", "sky surfing", "skydiving",
    "snow boarding", "snowmobile racing", "speed skating", "steer wrestling",
    "sumo wrestling", "surfing", "swimming", "table tennis", "tennis",
    "track bicycle", "trapeze", "tug of war", "ultimate", "uneven bars", "volleyball",
    "water cycling", "water polo", "weightlifting", "wheelchair basketball",
    "wheelchair racing", "wingsuit flying"
]

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to model's input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/")
def home():
    return render_template("index.html")  # Serve the frontend

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]

    return jsonify({"class": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
