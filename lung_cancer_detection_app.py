import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
from streamlit_space import space

st.set_page_config(layout="centered")


st.title("Lung Cancer Detection",)
space(lines=3)

# import Model_INCEPTION

@st.cache_resource
def load_model1():
    model = tf.keras.models.load_model("Model_INCEPTION.h5")
    return model

model=load_model1()

# fuction to get prediction
def import_and_predict(image_data, model):

    size = (299,299)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)


    return prediction

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Detect Cancer',
                          
                          ['Select Custom Image',
                           'Upload Testing Image',
                           ],
                          icons=['image','upload'],
                          default_index=0)




if selected == 'Upload Testing Image':
    file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    space()

    if file is None:
        st.text("Please upload an image file")
        space()
    else:
        image = Image.open(file)
        imgshow = image.resize((200, 200), Image.ANTIALIAS)
        st.image(imgshow)

        space()

        # st.image(image,use_column_width=True)
        detect_cancer = st.button("Detect cancer")
        if (detect_cancer):
            predictions = import_and_predict(image, model)
            class_names=['Lung adenocarcinoma','Lung benign tissue','Lung squamous cell carcinoma']

            val = max(predictions[0])*100
            strval = str(val)+' %'

            # val2 = np.argmax(predictions)

            print(predictions)
            print("predictions: ", predictions[0])
            string="This image most likely is : " +class_names[np.argmax(predictions)]
            # string2="The confidence of the prediction is: " , predictions[0][val2]
            string2="The confidence of this prediction is : " + strval

            space()
            st.success(string)
            st.success(string2)

if selected == 'Select Custom Image':
    file = st.sidebar.selectbox("Select Image", ("Test image 1.jpeg",
                                            "Test image 2.jpeg",
                                            "Test image 3.jpeg",
                                            ))

    st.subheader(file)
    space()
    image = Image.open(file)
    imgshow = image.resize((200, 200), Image.ANTIALIAS)
    st.image(imgshow)
    # st.image(image,use_column_width=True)

    space()
    detect_cancer = st.button("Detect cancer")
    if (detect_cancer):
        predictions = import_and_predict(image, model)
        class_names=['Lung adenocarcinoma','Lung benign tissue','Lung squamous cell carcinoma']

        val = max(predictions[0])*100
        strval = str(val)+' %'

        # val2 = np.argmax(predictions)
        
        print(predictions)
        print("predictions: ", predictions[0])
        string="This image most likely is : " +class_names[np.argmax(predictions)]
        # string2="The confidence of the prediction is: " , predictions[0][val2]
        string2="The confidence of this prediction is : " + strval

        st.success(string)
        st.success(string2)




