import os
import face_recognition as rec_face
import cv2
import numpy as np
from flask_socketio import SocketIO
from flask import Flask, render_template, Response 
import socket  # for getting the hostname and server address of the device running the script
import time
import functools
import base64
import uuid
import threading
import waitress

hostname=socket.gethostname() 
IPAddr=socket.gethostbyname(hostname) #Get Ip address
recogApp = Flask(__name__)   #Flask app initialized
socketioApp = SocketIO(recogApp) #socket IO app initialized

path = 'images'
images = []
image_names = []
myList = os.listdir(path)
print(myList)
print("http://"+IPAddr+":8080")

#Loop through folder of images and use cv.imread to read the images and then add them to a separate list
for img1 in myList:
    curImg = cv2.imread(f'{path}/{img1}')
    images.append(curImg)
    image_names.append(os.path.splitext(img1)[0].upper()) #get image names
print(image_names)

#encode list of images after inverting the color
def findEnc(images):
    encodeList = []
    for img2 in images:
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        encode = rec_face.face_encodings(img2)[0]
        # Add the encoding to the list
        encodeList.append(encode)
    return encodeList

# Define the font to be used for the text
font = cv2.FONT_HERSHEY_SIMPLEX

# Find the encodings for the images in the images list
encode_list_known = findEnc(images)

# Get a reference to webcam #0 (the default one)
capture = cv2.VideoCapture(0)

# Initialize variable videoFrame
global videoFrame, img

@functools.lru_cache(maxsize=None)
def recognize_faces():

     # Other required initializatons
    frame_count = 0
    start_time = time.time()

    while True:

        # Start/ Increment fps count at each loop
        frame_count += 1

        success, img = capture.read()

        # Reduce image size to increase speed
        image_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

        current_frame_faces = rec_face.face_locations(image_small)
        current_frame_encoding = rec_face.face_encodings(image_small, current_frame_faces)

        for encoded_face, face_location in zip(current_frame_encoding, current_frame_faces):
            matches = rec_face.compare_faces(encode_list_known, encoded_face)
            face_distance = rec_face.face_distance(encode_list_known, encoded_face)

            match_index = np.argmin(face_distance)

            # If a known face is detected
            if matches[match_index]:
                name = image_names[match_index].upper()

                y1, x2, y2, x1 = face_location
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                # GREEN Box with Name to visualise success confirmation
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # If an unknown face is detected
            else:

                y1, x2, y2, x1 = face_location
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                # RED Box with UNKNOWN to visualise error message
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255)) 
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "UNKNOWN", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', img)  # convert to jpg format for browser

        # Start the face detection and recognition thread
        thread = threading.Thread(target= recognize_faces)
        thread.start()

        #FPS COUNTER + Display on IMG Output
        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        # Draw FPS on the frame
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 475), font, 1, (255, 0, 0), 1, cv2.LINE_AA)        
        # Reset frame count and start time if elapsed time is greater than 1 second
        if elapsed_time > 1:
            frame_count = 0
            start_time = time.time()

        # output frames to client
        success, img_encoded = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()  # turn off cam
    cv2.destroyAllWindows()  # close all windows


# SocketIO event handler for saving the screenshot as an image
@socketioApp.on("save-screenshot")
def save_screenshot(data):
  # Get the image name and image data from the event data
  imageName = data["name"]
  img = data["image"]

  # Decode the image data (which is base64 encoded)
  image_decoded = base64.b64decode(img)

  # Save the image with the specified name in the "images" directory
  with open(f"images/{imageName}.jpg", "wb") as f:
    f.write(image_decoded)
  print(f"Saved image with name {imageName}.jpg")

# Define a route for the index page
@recogApp.route('/face_recognition_stream')
def face_recognition_stream():
    return Response(recognize_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

@recogApp.route('/')
def index():
    return render_template('index.html')

# Run the web application
def run():
    socketioApp.run(recogApp)

if __name__ == '__main__':
    socketioApp.run(recogApp)




# # Function to save the screenshot as an image with the specified name
# def saveScreenshot(imageName, img):
# # Encode the image as a base64 string
#     img_str = cv2.imencode('.jpg', img)[1].tostring()
#     img_b64 = base64.b64encode(img_str).decode()

#     # Generate a unique file name for the image
#     fileName = imageName + "-" + str(uuid.uuid4()) + ".jpg"

#     # Save the image to the "images" directory
#     with open(f"images/{fileName}", "wb") as f:
#         f.write(base64.decodebytes(img_b64.encode()))


# @socketioApp.on("save-screenshot")
# def saveScreenshot(data):
# # Get the image name from the message data 
#     imageName = data["name"]
# # Your code to save the screenshot as an image with the specified name goes here
#     cv2.imwrite(f"images/{imageName}.jpg", img)