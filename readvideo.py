import cv2
import numpy as np
import requests
import pytesseract #pip install tesseract
import os

from PIL import Image, ImageFont, ImageDraw
import winsound
import time
import psutil
import matplotlib.pyplot as plt


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" #Path to the tesseract 

######################subsetsum##########################

def display(lis):
    global risk
    global remarks
    remarks=''
    risk = 0
    for i in lis:
        risk+=dictonary.get(i)[1]
        remarks+=(dictonary.get(i)[0]+'\n')
    risk//=len(lis)
    print(risk)
    
def printSubsetsRec(arr,i,sum,lis):
    if(i==0 and sum!=0 and dp[0][sum]):
        lis.append(arr[i])
        display(lis)
        lis=[]
        return
    if(i==0 and sum==0):
        display(lis)
        lis=[]
        return
    if(dp[i-1][sum]):
        b=list(lis)
        printSubsetsRec(arr,i-1,sum,b)
    if(sum>=arr[i] and dp[i-1][sum-arr[i]]):
        lis.append(arr[i])
        printSubsetsRec(arr,i-1,sum-arr[i],lis)
        
    

def printSubsets(arr,n,sum):
    if n==0 or sum<0:
        return
    
    
    for i in range(n):
        for j in range(sum+1):
            dp[i][j]=False

    for i in range(n):
        dp[i][0]=True

    if arr[0]<=sum:
        dp[0][arr[0]]=True

    for i in range(1,n):
        for j in range(sum+1):
            dp[i][j] = (dp[i-1][j] or dp[i-1][j-arr[i]]) if(arr[i]<=j) else dp[i-1][j]

    if dp[n-1][sum]==False:
        print('No challans found')
        return 0

    
    printSubsetsRec(arr,n-1,sum,[])

    
    


#arr=[1,2,3,4,5]
#sum=10
dp=[]
def subsetSum(arr,n,sum):
    global dp
    dp=[[False for i in range(sum+1)] for j in range(len(arr))]
    printSubsets(arr,n,sum)
#printSubsets(arr,len(arr),sum)


#################end of subset sum#######################


def predictRisk(response):
    global dictonary
    dictonary={10000:['drunk&drive',10],100:['overloading',8],400:['over speeding',9],1000:['dangerous driving',9],500:['driving without license',6],1500:['driving without insurence',4],800:['signal jump',7],500:['no helmet',4],12000:['no permit',6],25000:['juvenile driving',9]}
    #print(dictonary)
    if type(response)!=dict:
        return 0
    if (type(response)==dict and (list(response.keys()).count('challanDetails') == 1 or len(response.keys())==0)):
        return 0
    amount = response['amount']
    subsetSum(list(dictonary.keys()),len(list(dictonary.keys())),amount)
    #return amount





def OCR(img):
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite("removed_noise.png", img)

    #  Apply threshold to get image with only black and white
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Write the image after apply opencv to do some ...

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img)

    # Remove template file
    #os.remove(temp)
    result=result.strip()
    result=result.replace(' ','')
    if result != '' and result!= None:
        #print(len(result))
        print(result)
        return result
    else:
        return ''

#numberPlateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
if __name__=='__main__':
    #show_img()
    global risk
    global remarks
    risk=0
    #URL='https://challan-api.herokuapp.com/'
    URL = 'https://api.apiclub.in/api/v1/challan_info/'

    video = cv2.VideoCapture('./test-7.mp4')

    if(video.isOpened()==False):
        print('Error Reading Video')

    while True:
        ret,frame = video.read()    
        gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plate = plat_detector.detectMultiScale(gray_video,scaleFactor=1.2,minNeighbors=5,minSize=(25,25))
        img=frame
        for (x,y,w,h) in plate:
            #cv2.imshow('plate',plate)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
            cv2.imshow('plate',frame[y:y+h,x:x+w])
            recognized_plate = OCR(frame[y:y+h,x:x+w])
            print(recognized_plate)
            #cv2.putText(frame,text='License Plate',org=(x-3,y-3),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,0,255),thickness=1,fontScale=0.6)
            #recognized_plate ='dl5sca6998' 
            PARAMS = {'number_plate' : recognized_plate}
            #URL+='dl5sca6998'

            response = requests.get(url = URL+recognized_plate, headers={'Referer':'157.48.160.115','API-KEY':'0da35e1a70835679a3104105473afaa7'})
            #response={'response':dict()}
            data = response.json()
            #print(data)
            #data=response
            predictRisk(data['response'])
            #print(type(img))
            # img=Image.fromarray(img)
            # d1 = ImageDraw.Draw(img)
            myFont = ImageFont.truetype('E:/PythonPillow/Fonts/arial.ttf', 40)
            if risk > 5:
                # d1.text((0, 0),'Vehicle number:'+recognized_plate+'\nPredicted Risk :'+str(risk)+'\nRemarks found :'+remarks, font=myFont, fill =(255, 0, 0))
                cv2.putText(img,text='Danger predicted \n Risk found is :'+str(risk),org=(20,20),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,0,255),thickness=2,fontScale=1.0)
                cv2.putText(img,text='risk is :'+str(risk),org=(x-3,y-3),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,0,255),thickness=2,fontScale=0.6)
                #cv2.imshow('Risk',img)

                winsound.Beep(600, 3000)
                #plt.pause(5)
                #cv2.destroyWindow('Risk')
            
        if ret == True:

            cv2.imshow('Video', frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()            
