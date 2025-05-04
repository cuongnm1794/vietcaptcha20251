from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_json
import base64
import time
import json
import ast


# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Danh sách các chữ cái
characters = ['1']

with open('char.txt', 'r') as file:
    file_contents = file.read()


characters = ast.literal_eval(file_contents)

# show log characters
print(characters)
    

# Kích thước hình ảnh resize
img_width = 320
img_height = 80

# Số lượng tối đa trong captcha ( dài nhất là 6)
max_length = 12

# convert chữ thành số
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)


# Convert số thành chữ
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


# Đọc ảnh base64 và mã hóa  
def encode_base64x(base64):
    img =  tf.io.decode_base64(base64)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    return {"image": img}
  
# Dịch từ mã máy thành chữ
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = []
    for res in results:
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
    return output_text

#load model japanese
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")


@app.route("/run",methods=["POST","GET"])
@cross_origin(origin='*')
def run():
  content = request.json
  start_time = time.time()
  imgstring =  content['base64'] 
  image_encode = encode_base64x(imgstring.replace("+","-").replace("/","_"))["image"]
  listImage = np.array([image_encode])
  preds = loaded_model.predict(listImage)
  pred_texts = decode_batch_predictions(preds)

  # remove - in pred_texts
  pred_texts = [x.replace('-', '') for x in pred_texts]

  return pred_texts[0].replace('[UNK]', '')


# Chạy server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868') #-> chú ý port, không để bị trùng với port chạy cái khác