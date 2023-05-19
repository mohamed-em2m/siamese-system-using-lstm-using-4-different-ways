from tensorflow import keras
import numpy as np
import librosa
from IPython.display import Audio
import joblib
#import sleep
from time import sleep
model = keras.saving.load_model(r'.\\cnn1d.h5')
model1 = keras.saving.load_model(r'.\\cnn2d.h5')
model2 = keras.saving.load_model(r'.\\lstm.h5')
labels = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    "follow",
    "forward",
    "backward",
    "learn",
    "visual"
]
def predict(audio):
    red,sr=librosa.load(audio,sr=22050)
    red=librosa.resample(red,orig_sr=sr,target_sr=22050)
    red=np.concatenate((np.zeros(max(22050-len(red),0)),red))
    #check if the audio is silent
    if red.max()<0.1:
        return "silence"
    #red = signal.medfilt(red, kernel_size=25)
    emam=librosa.feature.mfcc(y=red[:22050],n_mfcc=13).T
    label = np.zeros(40)
    
    r=np.argmax(model.predict(emam.reshape((1,44*13,1))),axis=1)
    r1 = np.argmax(model1.predict(emam.reshape((1,44,13,1))),axis=1)
    r2 = np.argmax(model2.predict(emam.reshape((1,44,13))),axis=1)
    encode = joblib.load(r'.\\encode')
    r=encode.inverse_transform(r)
    r1 = encode.inverse_transform(r1)
    r2 = encode.inverse_transform(r2)
    print(r,r1,r2)
    label[labels.index(r[0])]+=1
    label[labels.index(r1[0])]+=1
    label[labels.index(r2[0])]+=1
    if label.max()<2:
        return r2[0]
    r = labels[np.argmax(label)]
    return r

def record():
    #record audio from mic and save it as temp.wav
    import sounddevice as sd
    from scipy.io.wavfile import write
    fs = 22050  # Sample rate
    seconds = 1  # Duration of recording
    print("recording...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(r".\\temp.wav", fs, myrecording)  # Save as WAV file
    return r".\\temp.wav"
def realtime():
    #record audio from mic and pass it to predict function
    audio = record()
    prediction = predict(audio)
    print(prediction)
    return prediction

while True:
    
    pre = realtime()
    if pre == "stop":
        break
    sleep(1)
    