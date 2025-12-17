import os
import numpy as np
import cv2
import time
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ตั้งค่า Path พื้นฐานของโปรเจกต์
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# สร้างโฟลเดอร์เก็บรูปถ้ายังไม่มี
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# โหลดโมเดล Unet
MODEL_PATH = os.path.join(BASE_DIR, 'unet_best_bleeding_weights.h5')
model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    # 1. อ่านภาพแบบ Grayscale (0) ตามที่โมเดลต้องการ (1 Channel)
    img = cv2.imread(image_path, 0) 
    if img is None:
        raise ValueError(f"Could not read image at: {image_path}")
    
    # 2. ปรับขนาดเป็น 256x256 ตาม Expected Shape ของโมเดล
    img = cv2.resize(img, (256, 256)) 
    
    # 3. Normalization และเพิ่มมิติเป็น (1, 256, 256, 1)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  
    img = np.expand_dims(img, axis=-1) 
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # รับไฟล์จากชื่อ 'image_file' ใน HTML
        file = request.files.get('image_file')
        
        if file and file.filename != '':
            # บันทึกไฟล์รูปต้นฉบับ
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # ประมวลผลด้วยโมเดล Unet
            input_img = preprocess_image(image_path)
            prediction = model.predict(input_img)

            # จัดการผลลัพธ์ (เปลี่ยนจาก 0-1 เป็น 0-255)
            output_img = (prediction[0] * 255).astype(np.uint8)
            
            # บันทึกรูปผลลัพธ์ (Mask)
            output_filename = 'result_' + file.filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, output_img)

            # คำนวณพื้นที่พิกเซล (Area) สำหรับโชว์ใน UI
            pixel_count = np.sum(prediction > 0.5) 

            # ใช้ Timestamp เพื่อป้องกัน Browser Cache รูปเก่า
            ts = int(time.time())
            return render_template('index.html', 
                                   original_img=file.filename + "?t=" + str(ts),
                                   output_img=output_filename + "?t=" + str(ts),
                                   area=pixel_count)
    
    return render_template('index.html')

if __name__ == '__main__':
    # รันบนเครื่อง Port 5000
    app.run(port=5000, debug=True)
