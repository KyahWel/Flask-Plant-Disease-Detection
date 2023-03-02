
from flask import Flask, flash, request, redirect, url_for, Response, render_template
import cv2
import numpy as np
from keras.models import load_model
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("Credentials.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': 'https://test-8bf52-default-rtdb.firebaseio.com/'
})

app = Flask(__name__)
model = load_model("static\DiseaseDetector2.hdf5")

def allowed_file(filename):
    	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

ref = db.reference('Predictions/')
pred_ref = ref.child('camera1')

@app.route('/')
def index():
    return render_template('index.html')

def genCam1(cameraPort):
    cap = cv2.VideoCapture(cameraPort)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'
                   )

def genCam2(cameraPort):
    cap2 = cv2.VideoCapture(cameraPort)
    while True:
        _, frame2 = cap2.read()
        frame2 = cv2.flip(frame2, 1)
        ret, jpeg = cv2.imencode('.jpg', frame2)
        if ret:
            yield (b'--frame2\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'
                   )

def genCam3(cameraPort):
    cap3 = cv2.VideoCapture(cameraPort)
    while True:
        _, frame3 = cap3.read()
        frame3 = cv2.flip(frame3, 1)
        ret, jpeg = cv2.imencode('.jpg', frame3)
        if ret:
            yield (b'--frame3\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'
                   )

def genCam4(cameraPort):
    cap4 = cv2.VideoCapture(cameraPort)
    while True:
        _, frame4 = cap4.read()
        frame4 = cv2.flip(frame4, 1)
        ret, jpeg = cv2.imencode('.jpg', frame4)
        if ret:
            yield (b'--frame4\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'
                   )
            
def genCam2Predict(cameraPort):
    cap2 = cv2.VideoCapture(cameraPort)
    classes = ['Alternaria', 'Anthracnose', 'Downy Mildew', 'Healthy', 'Non-plants', 'White Rust']
    while True:
        _, frame2 = cap2.read()
        frame2 = cv2.flip(frame2, 1)
        img_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (224,224), interpolation = cv2.INTER_AREA)
        x = np.expand_dims(img_res , 0)
        pred = model.predict_on_batch(x).flatten()
        prediction = np.argmax(pred)

        if(cameraPort == 0):
            pred_ref = ref.child('camera1')
            pred_ref.update({
                'Class' : classes[int(prediction)],
                'PredMax' : int(prediction)
            })
        elif(cameraPort == 2):
            pred_ref = ref.child('camera2')
            pred_ref.update({
                'Class' : classes[int(prediction)],
                'PredMax' : int(prediction)
            })
        elif(cameraPort == 4):
            pred_ref = ref.child('camera3')
            pred_ref.update({
                'Class' : classes[int(prediction)],
                'PredMax' : int(prediction)
            })
        elif(cameraPort == 6):
            pred_ref = ref.child('camera4')
            pred_ref.update({
                'Class' : classes[int(prediction)],
                'PredMax' : int(prediction)
            })
        ret, jpeg = cv2.imencode('.jpg', frame2)
        print(prediction)
        if ret:
            yield (b'--frame2\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'
                   b'--predictions\r\n'
               b'Content-Type: application/json\r\n\r\n' + prediction.tobytes() + b'\r\n\r\n')


#Routes for multi camera rendering
@app.route('/video_feed0')
def video_feed0():
    return Response(genCam1(0), #change camera port
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    return Response(genCam2(2),  #change camera port
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(genCam3(4), #change camera port
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(genCam4(6),  #change camera port
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#Routes for camera prediction 
@app.route('/video_feedPredict0')
def video_feedPredict0():
    return Response(genCam2Predict(0),  #change camera port
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feedPredict1')
def video_feedPredict1():
    return Response(genCam2Predict(2),  #change camera port
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feedPredict2')
def video_feedPredict2():
    return Response(genCam2Predict(4),  #change camera port
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feedPredict3')
def video_feedPredict3():
    return Response(genCam2Predict(6),  #change camera port
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#Routes for single camera
@app.route('/camera_1')
def camera_1():
    return render_template('camera1.html')

@app.route('/camera_2')
def camera_2():
    return render_template('camera2.html')

@app.route('/camera_3')
def camera_3():
    return render_template('camera3.html')

@app.route('/camera_4')
def camera_4():
    return render_template('camera4.html')
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)