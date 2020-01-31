
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageDraw,Image,ImageFont


import numpy as np
from keras.optimizers import SGD
import cv2
import os
import time
import pandas as pd

num_classes=101
x=open('classInd.txt','r')
x1=x.read()
x2=list(x1.split('\n'))
x4=[]
for i in range(101):
    x3=x2[i].split(' ')[-1]
    x4.append(x3)
x5=[]
for i in range(101):
    x5.append(x4[i])
    print(x5)
#model = load_model('model-accuracy-99%.h5')
test_dir = r'E:\Nikhil\python\videoclass\videoclass\test'
test1_dir = r'C:\Users\Windows\Documents\action\humanactivity\test'
test2_dir=r'E:\Nikhil\python\ucf1\test'
test3_dir=r'E:\Nikhil\python\machinelearningex\videoclass\videoclass\test'
test4_dir=r'E:\ucf\test1'
ucf_dir=r'E:\ucf\dummy'
train1_data=r'C:\Users\Windows\Documents\action\humanactivity\train'
validation1_data=r'C:\Users\Windows\Documents\action\humanactivity\val'
test=r'E:\ucf\codes\testing'
cap=cv2.VideoCapture('floorgym.avi')
#img=cv2.imread(r'E:\ucf\codes\testing\test\test1.jpg')
img = Image.open(r"E:\ucf\codes\testing\test\test1.jpg")
font_type = ImageFont.truetype('arial.ttf',18)
#font_type1 = ImageFont.truetype('arial bold.ttf',20)
count=94
#font_type2 = ImageFont.truetype('arial italic.ttf',24)


while True:
    
    frameId=cap.get(100)
    ret,frame=cap.read()
    #cv2.imshow('fram1',frame)
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if(ret==True):
        
        
        #img1=cv2.resize(img,(1000,720))
        #cv2.imshow('output.jpg',img)
        filename=r'E:\ucf\codes\testing\test\test1.jpg'
        filename1=r'E:\ucf\codes\testing\test\test1.jpg'
        cv2.imwrite(filename,frame)
        #cv2.imwrite(filename1,img)
        #img2=cv2.imread(r'E:\ucf\codes\testing\test\test1.jpg')
        #cv2.imshow(r'E:\ucf\codes\testing\test\test1.jpg',img2)
        #model=Sequential()
        model=load_model('weights-improvement-01-0.86.hdf5')
        lrate=0.01
        epochs1=50
        decay=lrate/epochs1
        sgd = SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
        batch_size=1
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        test_datagen= ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            
            test,
            color_mode='grayscale',
            classes=['test'],
            target_size=(256,256),
            
            shuffle=False,
            batch_size=1
            )

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            
            horizontal_flip=True)
        train_generator = train_datagen.flow_from_directory(
            test2_dir,
            target_size=(224,224),
            batch_size=16,
            shuffle=False,
            classes=x5,
            )
            
        test_generator.reset()
        #train_generator.reset()
        #print(test_generator.filenames)
        #model1=model.predict_generator(test_generator,1200)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = test_datagen.flow_from_directory(
            test2_dir,
            target_size=(224,224),
            classes=x5,
            batch_size=1,
            )

        '''
        model.fit_generator(test_datagen,steps_per_epoch=5,epochs=1,verbose=0)
        scores=model.evaluate_generator(test_datagen,train_datagen)
        print(scores[0])
        '''



        pred= model.predict_generator(test_generator, test_generator.n//batch_size+1)
        predicted_class_indices=np.argmax(pred,axis=1)
        labels = (validation_generator.class_indices)
        labels2 = dict((v,k) for k,v in labels.items())
        predictions = [labels2[k] for k in predicted_class_indices]
        #print(predicted_class_indices)
        #print (labels)
                  
        #print (predictions)

        dt={}
        for i in range(len(test_generator.filenames)):
                
            name=test_generator.filenames[i].split('\\')[-1]
            dt[name]=predictions[i]
        img = Image.open(r"E:\ucf\codes\testing\test\test1.jpg")
        #img=cv2.imread(r'E:\ucf\codes\testing\test\test1.jpg')
        #cv2.putText(img,predictions[0], (10,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, 25)
        img1 = ImageDraw.Draw(img)
        img1.text((120,120),predictions[0], fill=(255),font=font_type)
        img.save(r'E:\ucf\codes\testing\framestovideo\test{}.jpg'.format(count))
        count=count+1
        
    #time.sleep(1)
        #print(dt)
    #cap.release()

'''
df1= pd.DataFrame(columns=['images','predicted'])
df1=pd.DataFrame(dt,index=['id'])
print(df1)
df1.to_csv('images.csv')

count=0
df=pd.read_csv('test1.csv',usecols=['predicted_output'])
for i in range(53101):
    if predictions[i]==df['predicted_output'][i]:
        count=count+1
accuracy=count/len(predictions)
print(accuracy)
'''
    

    

