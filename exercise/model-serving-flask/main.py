import numpy as np
import tensorflow as tf
from flask import Flask, request

MODEL_PATH = "fashion-mnist"
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
model = tf.keras.models.load_model(MODEL_PATH)
# ================Membuat Web App Sederhana Menggunakan Flask========================
app = Flask(__name__)


def preprocess_image(image):
    image = np.array(image) / 255.0
    image = np.expand_dims(image, 0)

    return image


# ===================================================================================


# ============Membuat API Endpoint dalam Web App sebagai Model Serving===============
@app.route("/")
def hello_world():
    return "Hello world"


@app.route("/predict", methods=["POST"])
def predict():
    request_json = request.json

    image = preprocess_image(request_json.get("data"))

    pred = model.predict(image)
    pred = tf.argmax(pred[0], -1).numpy()

    return {"prediction": CLASS_NAMES[pred]}


# ===================================================================================


# =============================Menjalankan Web App===================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
# ===================================================================================
