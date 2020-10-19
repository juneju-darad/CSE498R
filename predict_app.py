import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation,Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization,InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.applications import imagenet_utils

from flask import Flask, jsonify, request


import time
import cv2
from flask import Flask, render_template, Response

########################################################################################################################
########################################################################################################################
########################################################################################################################


# app = Flask(__name__)
#
#
#
# def get_model():
#     global model
#     model = Sequential()
#     model = tf.keras.models.load_model('asl3.h5')
#     print(' ** Model Loaded!')
#
# def preprocess_image(image, target_size):
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     image = image.resize(target_size)
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image = tf.keras.applications.mobilenet.preprocess_input(image)
#
#     return image
#
# print(' * Loading Keras model...')
# get_model()
#
#
#
# @app.route('/')
# def index():
#     """Video streaming home page."""
#     return render_template('predict.html')
#
#
# @app.route("/predict", methods=["POST"])
# def predict():
#     message = request.get_json(force=True)
#     encoded = message['image']
#     decoded = base64.b64decode(encoded)
#     image = Image.open(io.BytesIO(decoded))
#     preprocessed_image = preprocess_image(image, target_size=(64, 64))
#
#     prediction = model.predict(preprocessed_image)
#     x = np.argmax(prediction)
#     labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
#                    'M': 12,
#                    'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
#                    'Y': 24,
#                    'Z': 25, 'space': 26, 'del': 27, 'nothing': 28}
#     key = list(labels_dict.keys())
#     val = list(labels_dict.values())
#     x = key[val.index(x)]
#
#     response = {
#         'prediction' : {
#             'first' : x,
#             'second' : 'nothing'
#         }
#     }
#     return  jsonify(response)
#
#
# def gen():
#     """Video streaming generator function."""
#     cap = cv2.VideoCapture('Sample.MP4')
#
#     # Read until video is completed
#     while (cap.isOpened()):
#         # Capture frame-by-frame
#         ret, img = cap.read()
#         if ret == True:
#             img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
#             frame = cv2.imencode('.jpg', img)[1].tobytes()
#             yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             time.sleep(0.1)
#         else:
#             break
#
#
# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(gen(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# if __name__ == '__main__':
#     app.run(debug=True)


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


# app = Flask(__name__)
#
#
# def get_model():
#     global model
#     model = Sequential()
#     model = tf.keras.models.load_model('asl3.h5')
#     print(' ** Model Loaded!')
#
# def preprocess_image(image):
#
#
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image = tf.keras.applications.mobilenet.preprocess_input(image)
#
#     return image
#
# print(' * Loading Keras model...')
# get_model()
#
#
# @app.route('/')
# def index():
#     """Video streaming home page."""
#     return render_template('predict.html')
#
#
#
# def gen():
#     """Video streaming generator function."""
#     cap = cv2.VideoCapture('Sample2.MP4')
#     font = cv2.FONT_HERSHEY_SIMPLEX
#
#
#     # Read until video is completed
#     while (cap.isOpened()):
#         # Capture frame-by-frame
#         ret, img = cap.read()
#         if ret == True:
#             # frame = img.copy()
#             # cv2.rectangle(frame, (300,300),(50,50),(255,0,0),2)
#
#             img2 = cv2.resize(img, dsize=(64, 64))
#
#             preprocessed_image = preprocess_image(img2)
#
#             prediction = model.predict(preprocessed_image)
#             x = np.argmax(prediction)
#             labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
#                                'M': 12,
#                                'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
#                                'Y': 24,
#                                'Z': 25, 'space': 26, 'del': 27, 'nothing': 28}
#             key = list(labels_dict.keys())
#             val = list(labels_dict.values())
#             x = key[val.index(x)]
#             cv2.putText(img, x, (100, 100), font, 1, (255, 255, 0), 2)
#             cv2.rectangle(img, (300,300),(50,50),(255,0,0),2)
#
#             frame = cv2.imencode('.jpg', img)[1].tobytes()
#             yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             time.sleep(0.1)
#
#         else:
#             break
#
#
#
#
#
#
# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#
#


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera
from google.cloud import texttospeech


app = Flask(__name__)


video_stream = VideoCamera()


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = video_stream.get_frame()

        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break


@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/audio')
def audio():
    video_stream.__del__()
    video_stream.f.close()

    """Synthesizes speech from the input string of text or ssml.

    Note: ssml must be well-formed according to:
        https://www.w3.org/TR/speech-synthesis/

        #pip install --upgrade google-cloud-texttospeech
    """

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    with open('output.txt', 'r') as file:
        data = file.read().replace('\n', '')
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=data)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("static/output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

    return render_template('audio.html')


if __name__ == '__main__':
    app.run(debug=True)
