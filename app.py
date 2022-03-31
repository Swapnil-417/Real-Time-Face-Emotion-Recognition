import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

#Reading the model from JSON file
with open('model.json', 'r') as json_file:
    json_savedModel= json_file.read()

#load the model architecture 
classifier = model_from_json(json_savedModel)

# emotions 
emotion_dict = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

# load weights into new model
classifier.load_weights("final_model_weights.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
except Exception:
    st.write("Error loading cascade classifiers")


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # detecting multiple faces
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y-50), pt2=(
                x + w, y + h+10), color=(0, 255, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x+5, y-20)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return img



def main():
    # Title #
    st.title("Real Time Face Emotion Recognition")

    # activities
    activities = ["Home", "About", "Contact Us"]
    choice = st.sidebar.selectbox("Select from here", activities)

    st.markdown("<br>", unsafe_allow_html=True)


    if choice == "Home":

        st.markdown("<br>", unsafe_allow_html=True)

        html_temp1 = """<div style="background-color:#191970";padding:10px">
                                            <h4 style="color:White;text-align:center;">
                                            Head to the Sidebar to know more.</h4>
                                            </div>
                                            </br>"""

        #add paragraph text and position it in the center
        st.markdown("<p style='text-align:center'>Click on start to use your webcam</p>", unsafe_allow_html=True)


        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=VideoTransformer)


#       add space
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(html_temp1, unsafe_allow_html=True)

    elif choice == "About":

#         st.subheader("<p style='text-align:center'>About</p>")
        html_temp4 = """
                        <div style="background-color:#FFD700;padding:10px">
                        <h4 style="color:Black;text-align:center;">About</h4>
                        <h4 style="color:Black;text-align:center;">This Real Time Emotion Recognition web app is developed using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose only. If you want to contact us, just head to sidebar and click on Contact Us. </h4>
                        <h4 style="color:Black;text-align:center;">Developed with ❤ by Swapnil & Harish</h4>
                        </div>
                        <br></br>
                        """

        st.markdown(html_temp4, unsafe_allow_html=True)

    elif choice == "Contact Us":
        st.header("Contact Details")

        st.subheader("""Email""")
        st.info(""">* Swapnil Patil : patilswapn417@gmail.com
                    >* Harish Gawade : harishgawade199@gmail.com""")

        st.subheader(""" GitHub Profile""")     
        st.info(""">* [Swapnil Patil] (https://github.com/Swapnil-417) 
                           >* [Harish Gawade] (https://github.com/harishgawade1999)""")

        html_temp_copyright = """
                <div style="background-color:#DC143C ;padding:0.25px">
                        <h4 style="color:white;text-align:center;">Copyright © 2022 | Swapnil Patil </h4>
                        <h4 style="color:Black;text-align:center;">Thanks for Visiting ❤</h4>
                        </div>
                        <br></br>"""

        st.markdown(html_temp_copyright, unsafe_allow_html=True)


    else:
        pass
    
if __name__ == "__main__":
    main()
