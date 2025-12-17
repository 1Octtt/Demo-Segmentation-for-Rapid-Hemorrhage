import os
import numpy as np
import cv2
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- 1. การตั้งค่าเส้นทางและโฟลเดอร์ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 2. การจัดการโมเดล (GitHub Release) ---
# เปลี่ยนชื่อไฟล์เป็น v4 เพื่อให้ Render ล้างไฟล์ที่เคยโหลดเสียทิ้งไปครับ
MODEL_PATH = os.path.join(BASE_DIR, 'model_final_v4.h5')
# ลิงก์จาก GitHub Release v1.0 ของคุณ
MODEL_URL = 'https://github.com/1Octtt/Demo-Segmentation-for-Rapid-Hemorrhage/releases/download/v1.0/unet_best_bleeding_weights.h5'

def ensure_model_exists():
    """ตรวจสอบและดาวน์โหลดโมเดล หากไฟล์ไม่มีอยู่จริงหรือเสียหาย"""
    if not os.path.exists(MODEL_PATH):
        print("--- กำลังเริ่มดาวน์โหลดโมเดลสมบูรณ์จาก GitHub Release (360MB)... ---")
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print("--- ดาวน์โหลดโมเดลสำเร็จ! ---")
        except Exception as e:
            print(f"--- ดาวน์โหลดล้มเหลว: {e} ---")
            # ถ้าโหลดไม่สำเร็จ ให้ลบไฟล์ที่โหลดค้างไว้ทิ้ง เพื่อป้องกัน Error 'truncated file'
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            return False
    return True

# --- 3. ฟังก์ชันเตรียมรูปภาพก่อนวิเคราะห์ ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, 0) # อ่านรูปเป็น Grayscale
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# --- 4. หน้าเว็บหลักและการประมวลผล ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image_file')
        if file and file.filename != '':
            # บันทึกรูปภาพที่ผู้ใช้อัปโหลด
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # ตรวจสอบความพร้อมของโมเดล
            if ensure_model_exists():
                try:
                    # โหลดโมเดล (จะโหลดใหม่ทุกครั้งที่มีการกดวิเคราะห์ เพื่อความปลอดภัยของหน่วยความจำในแผน Free)
                    model = load_model(MODEL_PATH)
                    
                    input_img = preprocess_image(image_path)
                    if input_img is not None:
                        # ทำการวิเคราะห์ (Predict)
                        prediction = model.predict(input_img)
                        # แปลงผลลัพธ์กลับเป็นภาพสี (0-255)
                        output_img = (prediction[0] * 255).astype(np.uint8)

                        # บันทึกภาพผลลัพธ์
                        output_filename = 'result_' + file.filename
                        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                        cv2.imwrite(output_path, output_img)

                        return render_template('index.html', 
                                               original_img=file.filename,
                                               output_img=output_filename)
                except Exception as e:
                    return f"เกิดข้อผิดพลาดในการโหลดหรือรันโมเดล: {str(e)}"
            else:
                return "ไม่สามารถเข้าถึงไฟล์โมเดลได้ โปรดตรวจสอบการเชื่อมต่อ GitHub Release"
    
    return render_template('index.html')

# --- 5. การตั้งค่า Port สำหรับระบบ Cloud (Render) ---
if __name__ == '__main__':
    # ดึงค่า PORT จาก Environment Variable ของ Render (ปกติคือ 10000)
    port = int(os.environ.get("PORT", 10000))
    # ต้องใช้ host='0.0.0.0' เพื่อเปิดรับการเชื่อมต่อจากภายนอก
    app.run(host='0.0.0.0', port=port)
