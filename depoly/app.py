from flask import Flask, request, json, render_template

import pandas as pd
import pickle
import numpy as np
import random
import time
import tensorflow as tf
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array




class_mapping = open("inverse_class_mapping",'rb')
inverse_class_label = pickle.load(class_mapping)
class_mapping.close()


app = Flask(__name__)
def get_model():
    xception = tf.keras.applications.Xception(include_top=False,weights='imagenet',input_shape=(380,380,3))
    xception.trainable = False
    batch_norm = BatchNormalization()
    global_average_pool = GlobalAveragePooling2D()
    dense1 = Dense(256, activation='relu')
    dropout_layer = Dropout(0.5)
    dense2 = Dense(128, activation='relu')
    output =  Dense(120, activation='softmax')
    model = tf.keras.models.Sequential([
        xception,
        batch_norm,
        global_average_pool,
        dense1,
        dropout_layer,
        dense2,
        output
    ])
    model.load_weights("xception.h5")
    return model

model = get_model()

def get_model_pred(img):
    pred =  model.predict(img).argmax(axis=-1)
    breed = inverse_class_label[pred[0]]
    return breed

@app.route('/home')
def view():
    return render_template('pred.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Here")
    file = request.files['myfile']
    filename = file.filename
    img = load_img("../train/" +filename, target_size=(380, 380))
    img = img_to_array(img)
    img = np.true_divide(img, 255)
    img = np.expand_dims(img, axis=0)
    breed = get_model_pred(img) 
    
    return render_template('index.html', breed=breed)

@app.route('/')
def test():
    return "Server is running"

if __name__ == "__main__":
    app.run(debug=True)