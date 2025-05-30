from flask import Flask, request
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model("fire_model.h5")

def preprocess(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB").resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/upload', methods=['POST'])
def upload():
    if not request.data:
        return "No image received", 400
    image = preprocess(request.data)
    prediction = model.predict(image)[0][0]
    result = "FIRE" if prediction > 0.5 else "NO FIRE"
    print(f"Prediction: {result}")
    return result
