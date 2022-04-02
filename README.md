![Banner](https://github.com/Swapnil-417/Real-Time-Face-Emotion-Recognition/blob/main/data/Banner.png)


# Real-Time-Face-Emotion-Recognition
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

The Indian Education system has been undergoing rapid changes owing to the advancement of web-based learning services, specifically, eLearning platforms. In a physical classroom a lecturing teacher can see the faces and emotions of the students and tune their lecture accordingly whereas there are major challenges associated with digital learning. One of many challenges is how to ensure quality learning for students. In this project, our task is to build a real time emotion recognition system to monitor student's mood through class activity.

# Demo-Preview


https://user-images.githubusercontent.com/82964400/161369863-e2509fde-7ad1-404a-a3bb-f007a937e9ff.mp4



# Table of contents

- [Real-Time-Face-Emotion-Recognition](#real-time-face-emotion-recognition)
- [Demo-Preview](#demo-preview)
- [Table of contents](#table-of-contents)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Development](#model-development)
- [Deployment](#deployment)
- [Conclusion](#conclusion)
- [License](#license)

# Dependencies
[(Back to top)](#table-of-contents)

* TensorFlow
* OpenCV
* Streamlit
* Streamlit webRTC
* Heroku

# Installation
[(Back to top)](#table-of-contents)

To use this project, first clone the repo on your device using the command below:

```git init```

```git clone https://github.com/Swapnil-417/Real-Time-Face-Emotion-Recognition.git```

```cd Real-Time-Face-Emotion-Recognition```

Install dependencies and run app.py

```pip3 install -r requirements.txt```

```streamlit run app.py```

# Usage
[(Back to top)](#table-of-contents)

This webapp can help the teachers to analyze the students emotions and adjust their teaching strategies accordingly to improve the efficiency of online teaching.

# Model Development
[(Back to top)](#table-of-contents)

We build a simple Convolutional Neural Network model of Sequential type consisting of four Conv2D layers and two fully connected layers. The ReLU activation is used in layers. The other elements in the model structure are max pooling, batch normalization, dropout and flatten layer in between Convolution layers and FC layers. The output layer has 7 nodes of 7 emotions and softmax activation. We compiled the model using three parameters and used early stopping to halt the training at the right time. We trained the model setting the number of epochs to 30 and it stopped early after 13 epochs. We evaluated the model using the accuracy, categorical cross entropy loss and confusion matrix.

![Model Architecture](https://user-images.githubusercontent.com/82964400/161368890-04cc4deb-5881-4925-8a81-088daf116cdc.jpg)

## Model Evaluation

![Accuracy](https://github.com/Swapnil-417/Real-Time-Face-Emotion-Recognition/blob/main/results/final_cnn_accuracy.png)
![Loss](https://github.com/Swapnil-417/Real-Time-Face-Emotion-Recognition/blob/main/results/final_cnn_loss.png)

# Deployment
[(Back to top)](#table-of-contents)

For model deployment we created a front end using streamlit for web app and used streamlit-webrtc which helped us to deal with real time video streams. We used Haar-cascade for the detection of faces and OpenCV to read frames and image processing. We deployed this model on the streamlit cloud platform.

# Conclusion
[(Back to top)](#table-of-contents)

* Using a deep learning model based on the architecture of CNN, we developed a webapp to analyze student’s emotions. 
* The first model provided good accuracy but failed to detect emotions in real time. 
* The CNN model with deep layers was much more precise in detecting emotions in real time thus we deployed this model on streamlit cloud. 
* This webapp can help the teachers to analyze the students emotions and adjust their teaching strategies accordingly to improve the efficiency of online teaching.

# License
[(Back to top)](#table-of-contents)

[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)

