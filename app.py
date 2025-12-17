import os
import numpy as np
import cv2
import requests
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

MODEL_PATH = os.path.join(BASE_DIR, 'model_final_vFinal.h5')
MODEL_URL = 'https://github.com/1Octtt/Demo-Segmentation-for-Rapid-Hemorrhage/releases/download/v1.0/unet_best_bleeding_weights.h5'

# --- MODEL MANAGEMENT ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("--- Downloading Model (360MB)... Please wait. ---")
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("--- Download Complete! ---")
        except Exception as e:
            print(f"--- Download Error: {e} ---")
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)

# โหลดโมเดลครั้งเดียวตอนเริ่ม Server
download_model()
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("--- Model Loaded & Ready! ---")
    except Exception as e:
        print(f"--- Failed to load model: {e} ---")

# --- IMAGE PROCESSING ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, 0) # Read as Grayscale
    if img is None: return None
    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    original_img = None
    output_img = None
    
    if request.method == 'POST':
        file = request.files.get('image_file')
        if file and file.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            original_img = file.filename

            if model:
                input_img = preprocess_image(image_path)
                if input_img is not None:
                    prediction = model.predict(input_img)
                    # แปลงผลลัพธ์เป็นภาพขาวดำ
                    res_img = (prediction[0] * 255).astype(np.uint8)
                    
                    output_filename = 'res_' + file.filename
                    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), res_img)
                    output_img = output_filename

    return render_template('index.html', original_img=original_img, output_img=output_img)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
