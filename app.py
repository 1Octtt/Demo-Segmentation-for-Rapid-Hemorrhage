import os
import numpy as np
import cv2
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- 1. ตั้งค่าโฟลเดอร์เก็บไฟล์ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 2. ระบบดาวน์โหลดโมเดลจาก GitHub Release ---
# เปลี่ยนชื่อไฟล์เป็น v3 เพื่อบังคับให้ Render ดาวน์โหลดใหม่
MODEL_PATH = os.path.join(BASE_DIR, 'model_final_v3.h5')
# ใช้ลิงก์จาก GitHub Release ที่คุณกำลังสร้าง
MODEL_URL = 'https://github.com/1Octtt/Demo-Segmentation-for-Rapid-Hemorrhage/releases/download/v1.0/unet_best_bleeding_weights.h5'

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("--- กำลังดาวน์โหลดโมเดลจาก GitHub Release (360MB)... ---")
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("--- ดาวน์โหลดโมเดลสำเร็จ! ---")
        except Exception as e:
            print(f"--- ดาวน์โหลดล้มเหลว: {e} ---")

# --- 3. ฟังก์ชันเตรียมรูปภาพ ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, 0) # อ่านแบบ Gray scale
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# --- 4. หน้าเว็บหลัก ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image_file')
        if file and file.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # ตรวจสอบว่ามีโมเดลหรือยัง ถ้าไม่มีให้โหลด
            ensure_model_exists()
            
            try:
                model = load_model(MODEL_PATH)
                input_img = preprocess_image(image_path)
                
                if input_img is not None:
                    prediction = model.predict(input_img)
                    # แปลงผลลัพธ์เป็นภาพขาวดำ
                    output_img = (prediction[0] * 255).astype(np.uint8)

                    output_filename = 'result_' + file.filename
                    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                    cv2.imwrite(output_path, output_img)

                    return render_template('index.html', 
                                           original_img=file.filename,
                                           output_img=output_filename)
            except Exception as e:
                return f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}"
    
    return render_template('index.html')

# --- 5. การตั้งค่า Port สำหรับ Render ---
if __name__ == '__main__':
    # Render จะกำหนด Port ผ่าน Environment Variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
