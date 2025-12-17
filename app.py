import os
import numpy as np
import cv2
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- ตั้งค่าโฟลเดอร์เก็บไฟล์ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- ระบบดาวน์โหลดโมเดลอัตโนมัติจาก Google Drive ---
MODEL_PATH = os.path.join(BASE_DIR, 'unet_best_bleeding_weights.h5')
# ลิงก์ดาวน์โหลดตรงที่เตรียมไว้
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1YVRTnrhCDgZZPfEYWLjep14hMkABVst6'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("--- กำลังดาวน์โหลดโมเดลจาก Google Drive (330MB) โปรดรอสักครู่ ---")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("--- ดาวน์โหลดโมเดลสำเร็จ! ---")
        except Exception as e:
            print(f"--- เกิดข้อผิดพลาดในการดาวน์โหลด: {e} ---")

# เรียกใช้งานฟังก์ชันดาวน์โหลดก่อนเริ่มแอป
download_model()
model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = cv2.imread(image_path, 0) 
    if img is None:
        raise ValueError("ไม่สามารถอ่านรูปภาพได้")
    img = cv2.resize(img, (256, 256)) 
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image_file')
        if file and file.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            input_img = preprocess_image(image_path)
            prediction = model.predict(input_img)

            # แปลงผลลัพธ์เป็นภาพขาวดำ (Thresholding)
            output_img = (prediction[0] * 255).astype(np.uint8)

            output_filename = 'result_' + file.filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, output_img)

            return render_template('index.html', 
                                   original_img=file.filename,
                                   output_img=output_filename)
    
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
