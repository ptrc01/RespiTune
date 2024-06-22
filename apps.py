import os
import numpy as np
from flask import Flask, request, render_template, url_for
from keras.models import load_model
import librosa
from flask_mysqldb import MySQL


app = Flask(__name__, template_folder='Template', static_folder='static')
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flaskapp'

mysql = MySQL(app)
def stretch(data, rate):
    data = librosa.effects.time_stretch(data, rate=rate)
    return data

classes = ["COPD" ,"Bronchiolitis ", "Pneumoina", "URTI", "Healthy"]

gru_model1 = load_model('gru_model.h5')

def gru_diagnosis_prediction1(test_audio):
    data_x, sampling_rate = librosa.load(test_audio)
    data_x = stretch (data_x,1.2)

    features = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=52).T,axis = 0)

    features = features.reshape(1,52)

    test_pred = gru_model1.predict(np.expand_dims(features, axis = 1))
    classpreds = classes[np.argmax(test_pred[0], axis=1)[0]]
    confidence = test_pred.T[test_pred[0].mean(axis=0).argmax()].mean()

    print (classpreds , confidence)
    return classpreds, confidence


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/doctor')
def doctor():
     return render_template('doctor.html')

@app.route('/login')
def login():
     return render_template('login copy.html')

@app.route('/aboutus')
def aboutus():
     return render_template('aboutUs.html')

@app.route('/ambaturegister', methods=["POST"])
def ambaturegister():
     if request.method == 'POST':

            email = request.form.get('email')
            password = request.form.get('password')
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO user(email,password) VALUES(%s, %s)", (email, password))
            mysql.connection.commit()
            cur.close()
            return render_template("login copy.html", label="Your account was registered sucessfully")
     
     
@app.route('/ambatulogin', methods=["POST"])
def ambatulogin():

     #connection = mysql.connector.connect(host="localhost", user="root", password="", database="flaskapp")
     nocap = 1
     if request.method == 'POST':
        #try:
            email = request.form.get('email')
            password = request.form.get('password')
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM user WHERE email = %s AND password = %s", (email, password))
            user = cur.fetchone()
            cur.close()
            if user:
                 nocap = 2
                 #sessioning but i dont put sessions in this prototype
     
     if nocap == 1:
          return render_template("login copy.html", label="the user inquired does not exist")
     else:
          return render_template("Home.html")
     

     
@app.route('/profile')
def profile():
     return render_template('profile.html')

@app.route('/signup')
def signup():
     return render_template('sign-up.html')

@app.route('/pred')
def pred():
     return render_template('web1.html')

@app.route("/predict", methods=["POST"])
def predict():
   if "audioFile" not in request.files:
        return "No file part"

   if request.method == 'POST':
        #try:
            file = request.files['audioFile']

            #audio_data = file.read()

            sickness, confidences = gru_diagnosis_prediction1(file)
            print(f"Sickness: {sickness}, Confidence: {confidences}")
            return render_template("web2.html", sickness=sickness, confidence=confidences)
        
        #except Exception as e:
            #print(f"Error reading audio file: {e}")
            #return render_template("web2.html", sickness = 'OCPD', confidence = '0.8756')

if __name__ == "__main__":
    app.run(debug=True)
