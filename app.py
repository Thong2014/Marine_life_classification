import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Tiêu đề ứng dụng
st.header('Image Classification Model')

# Tải mô hình
model = load_model(r'C:\TLTN\Model\Marine_Life_classifier.keras')

# Danh sách các loại trái cây và rau củ
data_cat = [
    'Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Jelly Fish', 'Lobster', 
    'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffers', 'Sea Rays', 
    'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Shrimp', 'Squid', 'Starfish', 
    'Turtle_Tortoise', 'Whale'
]

# Kích thước ảnh đầu vào của mô hình
img_height = 180
img_width = 180

# Upload ảnh
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh đã tải lên
    st.image(uploaded_file, caption='Uploaded Image', width=200)
    
    # Xử lý ảnh
    image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)  # Chuyển ảnh thành mảng số
    img_bat = tf.expand_dims(img_arr, 0)  # Thêm batch dimension

    # Dự đoán
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])  # Chuẩn hóa xác suất
    
    # Hiển thị kết quả
    st.write('This in image is: **' + data_cat[np.argmax(score)] + '**')
    st.write('With accuracy of: **{:.2f}%**'.format(np.max(score) * 100))
