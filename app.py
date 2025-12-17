import os
import numpy as np
import cv2
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import threading
import time

app = Flask(__name__)

# --- 1. ตั้งค่าที่อยู่ไฟล์และโฟลเดอร์ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # จำกัด upload ที่ 50MB

# สร้างโฟลเดอร์เก็บรูปถ้ายังไม่มี
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 2. การจัดการโมเดล (GitHub Release) ---
MODEL_PATH = os.path.join(BASE_DIR, 'model_final_vFinal.h5')
MODEL_URL = 'https://github.com/1Octtt/Demo-Segmentation-for-Rapid-Hemorrhage/releases/download/v1.0/unet_best_bleeding_weights.h5'

# Global variable เก็บ state ของการดาวน์โหลด
model = None
model_loading = False
model_ready = False

def download_model_with_retry(max_retries=3):
    """ดาวน์โหลดโมเดลพร้อม retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"--- ครั้งที่ {attempt + 1}: กำลังดาวน์โหลดโมเดลจาก GitHub Release (360MB)... ---")
            
            response = requests.get(MODEL_URL, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"--- ความคืบหน้า: {progress:.1f}% ---")
            
            print("--- ดาวน์โหลดโมเดลสำเร็จ! ---")
            return True
            
        except Exception as e:
            print(f"--- ครั้งที่ {attempt + 1} ล้มเหลว: {e} ---")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"--- รอ {wait_time} วินาที ก่อนลองใหม่... ---")
                time.sleep(wait_time)
    
    return False

def ensure_model_exists():
    """ตรวจสอบและโหลดโมเดล"""
    global model, model_loading, model_ready
    
    if model_ready:
        return True
    
    if model_loading:
        return False
    
    if not os.path.exists(MODEL_PATH):
        model_loading = True
        print("--- เริ่มการดาวน์โหลดโมเดล ---")
        
        if download_model_with_retry():
            try:
                model = load_model(MODEL_PATH)
                model_ready = True
                model_loading = False
                print("--- โมเดลพร้อมใช้งาน ---")
                return True
            except Exception as e:
                print(f"--- ข้อผิดพลาดในการโหลดโมเดล: {e} ---")
                model_loading = False
                return False
        else:
            model_loading = False
            return False
    else:
        try:
            model = load_model(MODEL_PATH)
            model_ready = True
            return True
        except Exception as e:
            print(f"--- ข้อผิดพลาดในการโหลดโมเดล: {e} ---")
            return False

# --- 3. ฟังก์ชันเตรียมรูปภาพ (Preprocessing) ---
def preprocess_image(image_path):
    """เตรียมรูปภาพสำหรับ prediction"""
    try:
        img = cv2.imread(image_path, 0)
        if img is None:
            return None
        
        # ปรับขนาดภาพให้ตรงกับที่โมเดลต้องการ (256x256)
        img = cv2.resize(img, (256, 256))
        
        # Normalization
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        
        return img
    except Exception as e:
        print(f"--- ข้อผิดพลาดในการเตรียมรูปภาพ: {e} ---")
        return None

# --- 4. Health check ---
@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok', 'model_ready': model_ready}, 200

# --- 5. หน้าหลักและการประมวลผล ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image_file')
        
        if not file or file.filename == '':
            return render_template('index.html', error='กรุณาเลือกไฟล์รูปภาพ'), 400
        
        # ตรวจสอบนามสกุลไฟล์
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
            return render_template('index.html', error='รองรับเฉพาะไฟล์ PNG และ JPG'), 400
        
        # บันทึกรูปที่ผู้ใช้อัปโหลด
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # ตรวจสอบว่ามีโมเดลพร้อมใช้งานไหม
        if not model_ready:
            if not ensure_model_exists():
                if os.path.exists(image_path):
                    os.remove(image_path)
                return render_template('index.html', 
                                     error='เซิร์ฟเวอร์กำลังเตรียมโมเดล โปรดลองใหม่อีกครั้งในไม่กี่นาที'), 503
        
        try:
            # เตรียมภาพและทำ Prediction
            input_img = preprocess_image(image_path)
            
            if input_img is None:
                if os.path.exists(image_path):
                    os.remove(image_path)
                return render_template('index.html', 
                                     error='ไม่สามารถอ่านไฟล์รูปภาพได้'), 400
            
            # Prediction
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
            print(f"--- ข้อผิดพลาด: {str(e)} ---")
            if os.path.exists(image_path):
                os.remove(image_path)
            return render_template('index.html', 
                                 error=f'เกิดข้อผิดพลาดในการประมวลผล: {str(e)}'), 500
    
    return render_template('index.html')

# --- 6. ตั้งค่า Port สำหรับ Render ---
if __name__ == '__main__':
    # ดึงค่า PORT จากระบบของ Render
    port = int(os.environ.get("PORT", 10000))
    
    # เตรียมโมเดลตั้งแต่เริ่ม (ในแบบ background)
    print("--- เตรียมโมเดลตั้งแต่เริ่มต้น ---")
    threading.Thread(target=ensure_model_exists, daemon=True).start()
    
    # ต้องใช้ host='0.0.0.0' เพื่อให้โลกภายนอกเข้าถึงเว็บได้
    app.run(host='0.0.0.0', port=port, debug=False)
