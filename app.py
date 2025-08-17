'''
pip install flask tensorflow==2.19.0 keras==3.10.0

python app.py

http://127.0.0.1:5000

'''

import os
import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model

# This line suppresses tensorflow warnings, used in debugging due to py, tf, and keras mismatch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# This is the models path, take note that lambda was used
MODEL_PATH = "narrower_and_transfer_learning_model.keras"
model = load_model(MODEL_PATH, custom_objects={'Lambda': Lambda}, compile=False, safe_mode=False)

CLASS_NAMES = ['Benign', 'Cancer', 'Normal']

# This are templates and static folders according to flask app templates and examples
app = Flask(__name__, template_folder='templates', static_folder='static')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    # This right here preprocess uploaded image into (224,224,4) format
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224)) / 255.0
    mask = np.zeros_like(img)

    # This stacks 3 grayscale channels and a mask
    combined_input = np.stack([img, img, img, mask], axis=-1)
    return np.expand_dims(combined_input, axis=0).astype(np.float32)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", error="No file uploaded.")

        # Save uploaded file into static/uploads for data gathering and future model improvements if needed.
        filename = file.filename
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        input_tensor = preprocess_image(img_path)
        preds = model.predict(input_tensor, verbose=0)[0]

        # This section maps the results
        results = {CLASS_NAMES[i]: round(float(preds[i] * 100), 2) for i in range(len(CLASS_NAMES))}
        top_class = CLASS_NAMES[np.argmax(preds)]
        top_score = results[top_class]

        uploaded_image = f"{filename}"


        return render_template(
            "index.html",
            uploaded_image=uploaded_image,
            results=results,
            top_class=top_class,
            top_score=top_score
        )

    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)