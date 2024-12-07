!pip install flask-ngrok

!pip install pyngrok

import os
import threading
import json
from flask import Flask, render_template, request
from pyngrok import ngrok, conf
from pathlib import Path
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
from google.colab import drive
drive.mount('/content/drive')

# Đường dẫn đến mô hình đã được lưu trữ có độ chính xác cao nhất trên tập kiểm thử
model_path = '/content/drive/MyDrive/efficientnet/efficientnet/best_weights_model.keras'
best_model = load_model(model_path)

app = Flask(__name__)
port = "5000"

conf.get_default().auth_token = "1SdHSvXn625qjnKsUhD7UDVL3qE_7QW574BzWzFBUPc66BH9V"

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(port).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

urllib.request.urlretrieve("https://gist.githubusercontent.com/datpmwork/2aa0573436e5060f0a1066a69a98b180/raw/2adf06193d0e660ddfe21bf0957e6a6d88d591b8/data-model-uploader.html", "/content/uploader.html")

# ... Update inbound traffic via APIs to use the public-facing ngrok URL

# Define Flask routes
@app.route("/")
def index():
    return Path('/content/uploader.html').read_text()

@app.route("/upload_file", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        image_path = '/content/' + file.filename
        file.save(image_path)  # Save the file to a folder named 'uploads'

        # Đọc ảnh và chuyển về kích thước mong muốn (240x240 trong trường hợp này)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (240, 240))
        image = np.expand_dims(image, axis=0)  # Thêm chiều batch

        # Chuẩn hóa dữ liệu (nếu cần)
        # image = image / 255.0

        # Dự đoán nhãn
        prediction = best_model.predict(image)
        binary_prediction = np.round(prediction)

        return json.dumps(binary_prediction.tolist())

    return 'Error uploading file'

# Start the Flask server in a new thread
app.run(host='0.0.0.0', port=5000)