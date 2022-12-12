import requests, uuid
import matplotlib.image as mpimg

from flask import Flask, flash, request, redirect, url_for, render_template

from werkzeug.utils import secure_filename
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa
import librosa.display
import joblib
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import moviepy.editor as mp
app = Flask(__name__)

app.secret_key = "secret key"

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png','jpg'])
encoder=joblib.load('Encoder.joblib')
scaler=joblib.load('Scaler.joblib')
face_model = load_model('face_model.h5')
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
def pred_and_plot(model, filename, class_names):

    # Import the target image and preprocess it
    img = load_and_prep_image(filename)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    pred_class = class_names[int(tf.round(pred)[0][0])]

    return pred_class

def load_and_prep_image(filename, img_shape=48):

    # Read in target file (an image)
    img = tf.io.read_file(filename)

    # Decode the read file into a tensor & ensure 3 colour channels
    # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
    img = tf.image.decode_image(img, channels=1)

    # Resize the image (to the same size our model was trained on)
    img = tf.image.resize(img, size = [img_shape, img_shape])

    # Rescale the image (get all values between 0 and 1)
    img = img/255.
    return img

def noise(data):
    noise_amp = 0.04*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.70):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.8):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def higher_speed(data, speed_factor = 1.25):
    return librosa.effects.time_stretch(data, speed_factor)

def lower_speed(data, speed_factor = 0.75):
    return librosa.effects.time_stretch(data, speed_factor)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#sample_rate = 22050

def extract_features(data):

    result = np.array([])

    #mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=42) #42 mfcc so we get frames of ~60 ms
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    result = np.array(mfccs_processed)

    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=3, offset=0.5, res_type='kaiser_fast')

    #without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    #noised
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically

    #stretched
    stretch_data = stretch(data)
    res3 = extract_features(stretch_data)
    result = np.vstack((result, res3))

    #shifted
    shift_data = shift(data)
    res4 = extract_features(shift_data)
    result = np.vstack((result, res4))

    #pitched
    pitch_data = pitch(data, sample_rate)
    res5 = extract_features(pitch_data)
    result = np.vstack((result, res5))

    #speed up
    higher_speed_data = higher_speed(data)
    res6 = extract_features(higher_speed_data)
    result = np.vstack((result, res6))

    #speed down
    lower_speed_data = higher_speed(data)
    res7 = extract_features(lower_speed_data)
    result = np.vstack((result, res7))

    return result

def predict(path):
    features = get_features(path)
    check = []
    for elem in features:
        check.append(elem)
    total_model = load_model('model123.h5')
    ans = total_model.predict(np.expand_dims(scaler.transform(features),2))
    ans = ans.mean(axis=0)
    return ans

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and True:
        filename = secure_filename(file.filename)
#         m = pred_and_plot(face_model, file, class_names)
        cap = cv2.VideoCapture(filename)
        l = []
        # Check if camera opened successfully
        if (cap.isOpened()== False):
            print("Error opening video file")

        # Read until video is completed
        while(cap.isOpened()):

            # Capture frame-by-frame
            ret, frame = cap.read()

            if(True):
                if ret == True:
                    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grey = cv2.resize(grey, (48, 48))
                    l.append(face_model.predict(tf.expand_dims(grey, axis=0)))

                else:
                    break
        ans_1 = np.array(pd.DataFrame(np.squeeze(np.array(l))).mean(axis=0))
        ans_1 = np. delete(ans_1, 4)
        my_clip = mp.VideoFileClip(filename)
        my_clip.audio.write_audiofile("test.mp3")
        ans_2 = predict('test.mp3')
        ans = ans_1 + ans_2
        m = class_names[ans.argmax()]
        flash('The predicted emotion is: ' + m)
        return render_template('index.html')
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()
