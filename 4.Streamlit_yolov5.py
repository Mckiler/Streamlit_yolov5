#pip install streamlit

import shutil
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image

## เราสร้างฟังก์ชั่นโดยรับตัวแปรเป็น confident value เข้าไป ซึ่งจะมีค่าตั้งแต่ 0-1 ไว้สำหรับเป็นค่า threshold prediction
def get_yolov5(confident_val):    
   ## path ในที่นี้คือจะเป็น path ที่ไปยังโมเดลของเราที่วางไว้ครับ ถ้ามีชื่ออื่น จะต้องเปลี่ยนให้ตรงด้วย  
   model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
   model.conf = confident_val    
   return model
 
## สร้าง object ของโมเดลไว้สำหรับการเช็คโลโก้ในภาพ
model_logo = get_yolov5(0.4)

## Read Image
img = Image.open('test\9bf6d28723ca69e7.jpg')

## Inference โดย input คือรูปที่เราอ่านมา และใช้ขนาด 640
results = model_logo(img, size= 640)
results.render()

results.show()