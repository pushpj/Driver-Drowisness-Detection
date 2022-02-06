# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 17:57:16 2022

@author: Pushp jain
"""

import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import winsound

status = "Default value"

img_array = cv2.imread("D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\closed_eyes\s0001_00056_0_0_0_0_0_01.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img_array, cmap = "gray")
print(img_array.shape)

Datadirectory = "D:/DESKTOP/Data Science Projects/Drowsiness_Detection/Test_Dataset/"
Classes = ["closed_eyes", "open_eyes"]
for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        plt.imshow(img_array, cmap = "gray")
        plt.show()
        break
    break

img_size = 224
new_array = cv2.resize(backtorgb, (img_size, img_size))
plt.imshow(new_array, cmap = "gray")
plt.show()
print(new_array.shape)

# reading all images and converting them into an array for data and labels
training_data = []
def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                new_array = cv2.resize(backtorgb, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_Data()

print(len(training_data))

random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)

print(X.shape)

#Normalizing data
X = X/255.0;

Y = np.array(y)

#Saving data model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

# #Deep learning model for training and learning
# model = tf.keras.applications.mobilenet.MobileNet()

# #model.summary()

# #Using Transfer learning
# base_input = model.layers[0].input #Input

# base_output = model.layers[-4].output

# Flat_layer = layers.Flatten()(base_output)
# final_output = layers.Dense(1)(Flat_layer) #Only one node, either 0 or 1
# final_output = layers.Activation('sigmoid')(final_output)

# new_model = keras.Model(inputs = base_input, outputs = final_output)

# #new_model.summary()

# #Settings for binary classification (open/closed)
# new_model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# #new_model.fit(X, Y, epochs = 1, validation_split = 0.1) #Training the model

# new_model.save('my_model_3.h5')

new_model = tf.keras.models.load_model('mah_model.h5')

#Checking by predictions
# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\closed_eyes\s0001_00024_0_0_0_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("closed eye 1: ", prediction)


# ###############################################################################################################################################################################


# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\closed_eyes\s0001_00327_0_0_0_0_1_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("closed eye 2: ", prediction)


# ###############################################################################################################################################################################


# ###############################################################################################################################################################################


# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\closed_eyes\s0001_00485_0_0_0_0_1_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("closed eye 3: ", prediction)


# ###############################################################################################################################################################################


# ###############################################################################################################################################################################


# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\closed_eyes\s0001_00505_0_0_0_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("closed eye 4: ", prediction)


# ###############################################################################################################################################################################


# ###############################################################################################################################################################################


# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\closed_eyes\s0001_00918_0_0_0_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("closed eye 5: ", prediction)


# ###############################################################################################################################################################################


# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\closed_eyes\s0001_00735_0_0_0_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("closed eye 6: ", prediction)

# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\closed_eyes\s0001_00695_0_0_0_0_1_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("closed eye 7: ", prediction)

# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\closed_eyes\s0001_00515_0_0_0_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("closed eye 8: ", prediction)

# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\closed_eyes\s0001_00435_0_0_0_0_1_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("closed eye 9: ", prediction)

# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\closed_eyes\s0001_00245_0_0_0_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("closed eye 10: ", prediction)
























# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\open_eyes\s0001_01855_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("open eye 1: ", prediction)


# ###############################################################################################################################################################################


# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\open_eyes\s0001_01908_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("open eye 2: ", prediction)


# ###############################################################################################################################################################################


# ###############################################################################################################################################################################


# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\open_eyes\s0001_02031_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("open eye 3: ", prediction)


# ###############################################################################################################################################################################


# ###############################################################################################################################################################################


# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\open_eyes\s0001_02127_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("open eye 4: ", prediction)


# ###############################################################################################################################################################################


# ###############################################################################################################################################################################


# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\open_eyes\s0001_02257_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("open eye 5: ", prediction)


# ###############################################################################################################################################################################


# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\open_eyes\s0001_02084_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("open eye 6: ", prediction)

# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\open_eyes\s0001_02125_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)
# #20.8992009
# #26.3546411

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("open eye 7: ", prediction)

# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\open_eyes\s0001_01994_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("open eye 8: ", prediction)

# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\open_eyes\s0001_01854_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("open eye 9: ", prediction)

# img_array = cv2.imread('D:\DESKTOP\Data Science Projects\Drowsiness_Detection\Test_Dataset\open_eyes\s0001_01894_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
# backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
# new_array = cv2.resize(backtorgb, (img_size, img_size))

# X_input = np.array(new_array).reshape(1, img_size, img_size, 3)

# #print(X_input.shape)

# plt.imshow(new_array)

# X_input = X_input/255.0

# prediction = new_model.predict(X_input)
# print("open eye 10: ", prediction)




































































#Checking on random images
img = cv2.imread('D:/DESKTOP/Data Science Projects/Drowsiness_Detection/12.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

for(x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#Cropping of eye from image
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
for x, y, w, h in eyes:
    roi_gray = gray[y : y + h, x : x + w]
    roi_color = img[y : y + h, x : x + w]
    eyess = eye_cascade.detectMultiScale(roi_gray)
    if len(eyess) == 0:
        print("Eyes not detected!")
    else:
        for (ex, ey, ew, eh) in eyess:
            eyes_roi = roi_color[ey : ey + eh, ex : ex + ew]
            
plt.imshow(cv2.cvtColor(eyes_roi, cv2.COLOR_BGR2RGB))

print(eyes_roi.shape)

final_image = cv2.resize(eyes_roi, (224, 224))
final_image = np.expand_dims(final_image, axis = 0) #Needs fourth dimension
final_image = final_image/255.0

print(final_image.shape)

print(new_model.predict(final_image))

#Real time web cam video demo
frequency = 2500
duration = 1000 #in ms
path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
#Checking the web cam
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Error opening webcam!")

counter = 0
    
while True:
    ret, frame = cap.read()
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in eyes:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        eyess = eye_cascade.detectMultiScale(roi_gray)
        if len(eyess) == 0:
            print("Eyes not detected!")
        else:
            for (ex, ey, ew, eh) in eyess:
                eyes_roi = roi_color[ey : ey + eh, ex : ex + ew]
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(faceCascade.empty())
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    
    #To draw a rectange around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    final_image = cv2.resize(eyes_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis = 0) #Needs fourth dimension
    final_image = final_image/255.0
    
    Predictions = new_model.predict(final_image)
    print(Predictions)
    if (Predictions > [[21.124195]]):
        status = "Open eyes"
        cv2.putText(frame, status, (150, 150), font, 3, (0, 255, 0), 2, cv2.LINE_4)
        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, 'Active', (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        counter = counter + 1
        status = "Closed eyes"
        cv2.putText(frame, status, (150, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if counter > 5:
            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            cv2.putText(frame, 'Sleep Alert!!!', (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            winsound.Beep(frequency, duration)
            counter = 0
    
    #For putting text on vdo -> putText() method
    #cv2.putText(frame, status, (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    cv2.imshow('Drowsiness Detection', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()



print('Execution Completed!')