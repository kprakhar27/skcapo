from flask import Flask, flash, request, redirect, url_for, render_template
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from PIL import Image
import os

# imports

from keras.models import model_from_json 

# opening and store file in a variable

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model

loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

label=['Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanocytic nevi',
    'Vascular lesions',
    'Melanoma']

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Set up the main route
from flask import send_from_directory

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory('', name)

@app.route('/', methods=['GET', 'POST'])

def main():
    if request.method == 'GET':
        return(render_template('index2.html'))
            
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('', filename))
            x = np.asarray(Image.open(filename).resize((100,75)))
            img = x.reshape((1,75,100,3))
            out = loaded_model.predict(img)
            prob = max(out[0])
            for i in np.argmax(out,axis=1):
                cancer_type = label[i]
            print(prob, cancer_type)
        os.remove(filename)
        return render_template('predict.html', cancer = cancer_type, probability = prob)

if __name__ == '__main__':
    app.run()
