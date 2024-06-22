from keras.models import load_model
import os
import numpy as np
import librosa

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

gru_diagnosis_prediction1('D:/Visual Studio Code/AI Project/input/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/226_1b1_Al_sc_Meditron.wav')
