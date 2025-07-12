from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
import pickle
from sklearn.model_selection import train_test_split
from keras.applications import ResNet101
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, InputLayer, BatchNormalization, Dropout

main = tkinter.Tk()
main.title("Paddy Crop Disease Detection using Ann, CNN & Resnet101")
main.geometry("1300x1200")

global X, Y, filename, X_train, X_test, y_train, y_test, resnet_model
labels = ['Blast', 'BrownSpot', 'Hispa', 'Normal']
global accuracy, precision, recall, fscore

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    sourcelabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" dataset loaded\n\n")
    Y = np.load('model/Y.txt.npy')
    unique, count = np.unique(Y, return_counts = True)
    height = count
    bars = ('Leaf Blast','Brown Spot','Hispa', 'Healthy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Disease Names")
    plt.ylabel("Images Count")
    plt.title("Different Diseases found in Dataset")
    plt.show()

def processDataset():
    global X, Y, filename, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32, 32))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(32, 32, 3)
                    X.append(im2arr)
                    label = getID(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Total Images found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset Train & Test Split Details\n\n")
    text.insert(END,"80% dataset used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset used for training : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    test = X[3]
    cv2.imshow("Sample Processed Image",cv2.resize(test,(150,250)))
    cv2.waitKey(0)

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    labels = ['Leaf Blast', 'BrownSpot', 'Hispa', 'Healthy']    
    conf_matrix = confusion_matrix(testY, predict) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def runANN():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], (X_train.shape[2] * X_train.shape[3])))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], (X_test.shape[2] * X_test.shape[3])))
    ann_model = Sequential()
    ann_model.add(Flatten(input_shape=[X_train1.shape[1], X_train1.shape[2]]))
    ann_model.add(Dense(30, activation="relu"))
    ann_model.add(Dense(10, activation="relu"))
    ann_model.add(Dense(y_train.shape[1], activation="softmax"))
    ann_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/ann_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/ann_weights.hdf5', verbose = 1, save_best_only = True)
        hist = ann_model.fit(X_train1, y_train, batch_size = 16, epochs = 80, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)          
    else:
       ann_model.load_weights("model/ann_weights.hdf5")
    predict = ann_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("ANN Algorithm", y_test1, predict)
       
def runCNN():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    cnn_model = Sequential()
    cnn_model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    cnn_model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    cnn_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    cnn_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    cnn_model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=100, activation='relu'))
    cnn_model.add(Dense(units=100, activation='relu'))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Dense(units=y_train.shape[1], activation='softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 16, epochs = 80, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)        
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("CNN Algorithm", y_test1, predict)    

def runResnet():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore, resnet_model
    resnet = ResNet101(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in resnet.layers:
        layer.trainable = False
    resnet_model = Sequential()
    resnet_model.add(resnet)
    resnet_model.add(Convolution2D(32, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (1, 1)))
    resnet_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (1, 1)))
    resnet_model.add(Flatten())
    resnet_model.add(Dense(units = 256, activation = 'relu'))
    resnet_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    print(resnet_model.summary())
    resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/resnet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = resnet_model.fit(X_train, y_train, batch_size = 16, epochs = 80, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    else:
        resnet_model = load_model("model/resnet_weights.hdf5")
    predict = resnet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("ResNet Algorithm", y_test1, predict)

def graph():
    df = pd.DataFrame([['ANN','Precision',precision[0]],['ANN','Recall',recall[0]],['ANN','F1 Score',fscore[0]],['ANN','Accuracy',accuracy[0]],
                       ['CNN','Precision',precision[1]],['CNN','Recall',recall[1]],['CNN','F1 Score',fscore[1]],['CNN','Accuracy',accuracy[1]],
                       ['ResNet','Precision',precision[2]],['ResNet','Recall',recall[2]],['ResNet','F1 Score',fscore[2]],['ResNet','Accuracy',accuracy[2]],                   
                  ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def predict():
    global resnet_model
    labels = ['Leaf Blast', 'BrownSpot', 'Hispa', 'Healthy'] 
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32, 32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = resnet_model.predict(img)
    predict = np.argmax(preds)
    max_value = np.amax(preds)
    if max_value > 0.95:
        img = cv2.imread(filename)
        img = cv2.resize(img, (700,400))
        cv2.putText(img, 'Paddy Disease Predicted as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.imshow('Paddy Disease Predicted as : '+labels[predict], img)
        cv2.waitKey(0)
    else:
        img = cv2.imread(filename)
        img = cv2.resize(img, (700,400))
        cv2.putText(img, 'Not a paddy leaf image', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.imshow('Not a paddy leaf image', img)
        cv2.waitKey(0)

font = ('times', 14, 'bold')
title = Label(main, text='Paddy Crop Disease Detection using Ann, CNN & Resnet101')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Paddy Disease Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

sourcelabel = Label(main)
sourcelabel.config(bg='brown', fg='white')  
sourcelabel.config(font=font1)           
sourcelabel.place(x=460,y=100)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1)

annButton = Button(main, text="Run ANN Algorithm", command=runANN)
annButton.place(x=50,y=200)
annButton.config(font=font1) 

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN)
cnnButton.place(x=350,y=200)
cnnButton.config(font=font1)

resnetbutton = Button(main, text="Run Resnet Algorithm", command=runResnet)
resnetbutton.place(x=600,y=200)
resnetbutton.config(font=font1) 

graphbutton = Button(main, text="Comparison Graph", command=graph)
graphbutton.place(x=50,y=250)
graphbutton.config(font=font1)

predictbutton = Button(main, text="Predict Disease from Test Image", command=predict)
predictbutton.place(x=350,y=250)
predictbutton.config(font=font1) 

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
