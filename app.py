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

# --- 2. ตั้งค่าโมเดล ---
MODEL_PATH = os.path.join(BASE_DIR, 'model_final_vFinal.h5')
MODEL_URL = 'https://github.com/1Octtt/Demo-Segmentation-for-Rapid-Hemorrhage/releases/download/v1.0/unet_best_bleeding_weights.h5'

def download_model_if_not_exists():
    """ฟังก์ชันเช็คและดาวน์โหลดโมเดล"""
    if not os.path.exists(MODEL_PATH):
        print("--- ไม่พบโมเดล! กำลังเริ่มดาวน์โหลด (360MB)... ---")
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
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            return False
    return True

# --- 3. โหลดโมเดลขึ้น RAM (ทำครั้งเดียวตอน Start Server) ---
# เราเรียกดาวน์โหลดก่อน แล้วค่อย Load เข้าสู่ตัวแปร model
model = None
if download_model_if_not_exists():
    try:
        print("--- กำลังโหลดโมเดลเข้าหน่วยความจำ (RAM)... ---")
        model = load_model(MODEL_PATH)
        print("--- โหลดโมเดลสำเร็จและพร้อมใช้งาน! ---")
    except Exception as e:
        print(f"--- โหลดโมเดลไม่สำเร็จ: {e} ---")

# --- 4. ฟังก์ชันเตรียมรูปภาพ ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# --- 5. Routes สำหรับหน้าเว็บ ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image_file')
        if file and file.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            if model is not None:
                try:
                    input_img = preprocess_image(image_path)
                    if input_img is not None:
                        prediction = model.predict(input_img)
                        output_img = (prediction[0] * 255).astype(np.uint8)

                        output_filename = 'result_' + file.filename
                        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                        cv2.imwrite(output_path, output_img)

                        return render_template('index.html', 
                                               original_img=file.filename,
                                               output_img=output_filename)
                except Exception as e:
                    return f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}"
            else:
                return "เซิร์ฟเวอร์ยังโหลดโมเดลไม่เสร็จ หรือไฟล์โมเดลมีปัญหา"
    
    return render_template('index.html')

# --- 6. การตั้งค่า Port สำหรับ Render ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
