import streamlit as st
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from cv2 import cv2 # from cv2 사용
import os

import tensorflow as tf
import keras
from keras.models import load_model
keras.backend.backend(), keras.__version__

# 이미지 차원과 비디오 입력 프레임 정의
image_height, image_width = 224, 224
input_frames = 32



# mp4 비디오 파일을 모델 입력에 맞게 수정하는 함수
def preprocess_video(input_file, num_frames=input_frames):
    '''
    input_file: 업로드한 파일 그대로
    '''
    frames = []

    vid = input_file.name
    with open(vid, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(vid)
    stframe = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)  # 변경된 부분

    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if not ret:
            st.text("Can't receive frame")
            break
        # st.text('frame: {}'.format(index))
        stframe.image(np.asarray(frame), channels='BGR')
        frame = cv2.resize(frame, (image_width, image_height))
        frames.append(frame)

    cap.release()

    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=0)

    return frames

# 예측값을 처리해서 바꾸는 함수
def predict_ratioA(prediction):
    # 클래스를 나타내는 라벨 ["class" : ratioA]
    class_labels = {"0" : 0, "1" : 20, "2" : 30, "3" : 40, "4" : 50, "5" : 60, "6" : 70, "7" : 80, "8" : 100}

    probabilities = tf.nn.softmax(prediction)
    probabilities = probabilities.numpy().squeeze(0)

    # 예측된 클래스
    predicted_class = np.argmax(probabilities)
    predicted_ratioA = class_labels[str(predicted_class)]

    print("Predicted ratioA:", predicted_ratioA)
    print("Probability:", probabilities[predicted_class])
    print("")

    return predicted_ratioA


# 웹 제목 및 업로드 박스
st.title('Capstone Design')
uploaded_file = st.file_uploader('영상을 선택해주세요.', type=['mp4'])
vid_area = st.empty()

if uploaded_file is None:
    st.text('파일을 먼저 올려주세요.')
else:
    # 모델 사용하기
    
    X = preprocess_video(uploaded_file) # uploaded_file) # mp4 파일을 입력으로 변형


results = ''

st.success('results')
