import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# import Model_INCEPTION

def load_model1():
    model = tf.keras.models.load_model("Model_INCEPTION.h5")
    return model

model=load_model1()


st.set_page_config(layout="wide")
st.title("Lung Cancer Detection")

file = st.file_uploader("Please upload an image", type=["png", "jpg"])

def import_and_predict(image_data, model):

    size = (299,299)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)


    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    imgshow = image.resize((200, 150), Image.ANTIALIAS)
    st.image(imgshow)
    # st.image(image,use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names=['Lung adenocarcinoma','Lung benign tissue','Lung squamous cell carcinoma']

    val = max(predictions[0])
    strval = str(val)

    # val2 = np.argmax(predictions)

    print(predictions)
    print("predictions: ", predictions[0])
    string="This image most likely is: "+class_names[np.argmax(predictions)]
    # string2="The confidence of the prediction is: " , predictions[0][val2]
    string2="The confidence of the prediction is: " + strval

    st.success(string)
    st.success(string2)





