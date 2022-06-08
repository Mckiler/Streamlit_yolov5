#pip install streamlit

import shutil
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
from io import BytesIO

from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## เราสร้างฟังก์ชั่นโดยรับตัวแปรเป็น confident value เข้าไป ซึ่งจะมีค่าตั้งแต่ 0-1 ไว้สำหรับเป็นค่า threshold prediction
def get_yolov5(confident_val):    
   ## path ในที่นี้คือจะเป็น path ที่ไปยังโมเดลของเราที่วางไว้ครับ ถ้ามีชื่ออื่น จะต้องเปลี่ยนให้ตรงด้วย  
   model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
   model.conf = confident_val    
   return model

 
st.sidebar.image("logo.png", caption=None, width=300, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
st.sidebar.header('👉 Mckiler 👈')
st.sidebar.header('Choose Prediction Model')
model_choice = st.sidebar.selectbox('Logo Detection Model', ['YoloV5s Prediction','Other predictions'], key='1')

if model_choice == 'YoloV5s Prediction':
    st.subheader("Image")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    
    if image_file is not None:
    
        # To See details
        file_details = {"filename":image_file.name, "filetype":image_file.type,
                        "filesize":image_file.size}
        # st.write(file_details)
 
        st.image(image_file, caption='Image', channels='RGB',output_format='auto')
     
        ## สร้าง object ของโมเดลไว้สำหรับการเช็คโลโก้ในภาพ
        model_logo = get_yolov5(0.4)
        
        ## Read Image
        # img = Image.open('test\9bf6d28723ca69e7.jpg')
        img = Image.open(image_file)

        ## Inference โดย input คือรูปที่เราอ่านมา และใช้ขนาด 640
        results = model_logo(img, size= 640)
        results.render()
        for img in results.imgs:
            buffered = BytesIO()
            img_base64 = Image.fromarray(img)
  
        st.image(img_base64, caption='Image', channels='RGB',output_format='auto')
        # results.show()