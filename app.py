import os
import numpy as np
import cv2
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- 1. ตั้งค่าที่อยู่ไฟล์และโฟลเดอร์ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# สร้างโฟลเดอร์เก็บรูปถ้ายังไม่มี
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 2. การจัดการโมเดล (GitHub Release) ---
# [cite_start]เปลี่ยนชื่อไฟล์เป็น vFinal เพื่อให้ Render ล้างไฟล์เก่าที่เคยโหลดเสีย (HTML) ทิ้งไป [cite: 1]
MODEL_PATH = os.path.join(BASE_DIR, 'model_final_vFinal.h5')
# ลิงก์ตรงจาก Assets ใน GitHub Release v1.0 ของคุณ
MODEL_URL = 'https://github.com/1Octtt/Demo-Segmentation-for-Rapid-Hemorrhage/releases/download/v1.0/unet_best_bleeding_weights.h5'

def ensure_model_exists():
    """ฟังก์ชันตรวจสอบและดาวน์โหลดโมเดลจาก GitHub"""
    if not os.path.exists(MODEL_PATH):
        print("--- กำลังเริ่มดาวน์โหลดโมเดลจาก GitHub Release (360MB)... ---")
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print("--- ดาวน์โหลดโมเดลสำเร็จ! ---")
            return True
        except Exception as e:
            print(f"--- ดาวน์โหลดล้มเหลว: {e} ---")
            # ถ้าโหลดพลาด ให้ลบไฟล์ทิ้งเพื่อไม่ให้เกิด truncated file error ในครั้งหน้า
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            return False
    return True

# --- 3. ฟังก์ชันเตรียมรูปภาพ (Preprocessing) ---
def preprocess_image(image_path):
    # อ่านภาพเป็น Grayscale (0)
    img = cv2.imread(image_path, 0)
    if img is None:
        return None
    # ปรับขนาดภาพให้ตรงกับที่โมเดลต้องการ (256x256)
    img = cv2.resize(img, (256, 256))
    # ทำ Normalization
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# --- 4. หน้าหลักและการประมวลผล ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image_file')
        if file and file.filename != '':
            # บันทึกรูปที่ผู้ใช้อัปโหลด
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # ตรวจสอบว่ามีโมเดลพร้อมใช้งานไหม
            if ensure_model_exists():
                try:
                    # โหลดโมเดล
                    model = load_model(MODEL_PATH)
                    
                    # เตรียมภาพและทำ Prediction
                    input_img = preprocess_image(image_path)
                    if input_img is not None:
                        prediction = model.predict(input_img)
                        # แปลงผลลัพธ์กลับเป็นภาพ 0-255
                        output_img = (prediction[0] * 255).astype(np.uint8)

                        # บันทึกภาพ Prediction
                        output_filename = 'result_' + file.filename
                        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                        cv2.imwrite(output_path, output_img)

                        return render_template('index.html', 
                                               original_img=file.filename,
                                               output_img=output_filename)
                except Exception as e:
                    return f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}"
            else:
                return "เซิร์ฟเวอร์กำลังเตรียมโมเดล โปรดลองใหม่อีกครั้งใน 1 นาที"
    
    return render_template('index.html')

# --- 5. การตั้งค่า Port สำหรับ Render ---
if __name__ == '__main__':
    # ดึงค่า PORT จากระบบของ Render (ปกติคือ 10000)
    port = int(os.environ.get("PORT", 10000))
    # ต้องใช้ host='0.0.0.0' เพื่อให้โลกภายนอกเข้าถึงเว็บได้
    app.run(host='0.0.0.0', port=port)
