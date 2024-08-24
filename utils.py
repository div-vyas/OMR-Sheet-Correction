import cv2
import numpy as np

##############################################################################
def stackImages(imgArray,scale,labels=[]):
    rows=len(imgArray)
    cols=len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0],list)
    width=imgArray[0][0].shape[1]
    height=imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0,rows):
            for y in range(0,cols):
                imgArray[x][y]=cv2.resize(imgArray[x][y],(0,0),None,scale,scale)
                if len(imgArray[x][y].shape)==2: imgArray[x][y]=cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
        imageBlank=np.zeros((height,width,3),np.uint8)
        hor=[imageBlank]*rows
        hor_con = [imageBlank] * rows
        for x in range(0,rows):
            hor[x]=np.hstack(imgArray[x])
            hor_con[x]=np.concatenate(imgArray[x])
        ver=np.vstack(hor)
        ver_con=np.concatenate(hor)
    else:
        for x in range(0,rows):
            imgArray[x]=cv2.resize(imgArray[x],(0,0),None,scale,scale)
            if len(imgArray[x].shape)==2: imgArray[x]=cv2.cvtColor(imgArray[x],cv2.COLOR_GRAY2BGR)
        hor=np.hstack(imgArray)
        hor_con=np.concatenate(imgArray)
        ver=hor
    if len(labels)!=0:
        eachImgWidth=int(ver.shape[1]/cols)
        eachImgHeight=int(ver.shape[0]/rows)
        #print(eachImgHeight)
        for d in range(0,rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(labels[d][c])*13+27,30+eachImgHeight*d))
                cv2.putText(ver,labels[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)

    return ver
##################################################################################

def rectContour(contours):

    rectCon=[]
    for i in contours:
        area = cv2.contourArea(i)
        #print("Area",area)
        if area>50:
            peri=cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.02*peri,True)
            #print("Corner points",len(approx))
            if len(approx)==4:
                rectCon.append(i)
    rectCon=sorted(rectCon,key=cv2.contourArea,reverse=True)

    return rectCon

###################################################################################

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return (approx)

##################################################################################

def reorder(myPoints):
    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2),np.int32)
    add=myPoints.sum(1)
    #print(myPoints)
    #print(add)
    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[3]= myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print (diff)

    return myPointsNew

#################################################################################

def splitBoxes(img):
    rows=np.vsplit(img,5)
    boxes=[]
    for r in rows:
        cols=np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
            cv2.imshow("Split",box)
    return boxes

#################################################################################

def showAnswers(img,myIndex,grading,ans,questions,choices):
    secW = int(img.shape[1] / questions)
    secH = int(img.shape[0] / choices)

    for x in range(0,questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW//2
        cY = (x * secH) + secH//2

        if grading[x]==1:
            myColor=(0,255,0)
        else:
            myColor=(0,0,255)
            correctAns = ans[x]
            cv2.circle(img, ((correctAns * secW)+secW//2,(x*secH)+secH//2), 50, (0,255,0), cv2.FILLED)

        cv2.circle(img,(cX, cY) ,50,myColor,cv2.FILLED)

    return img


#################################################################################

#1. `stackImages(imgArray, scale, labels=[]):`
   #- This function takes three parameters:
    # - `imgArray`: A list of lists of images. It can be a 2D list where each element is an image.
     #- `scale`: A scaling factor to resize the images in the `imgArray`.
     #- `labels`: An optional list of labels for the images.
   #- It first determines the dimensions of the `imgArray`, checks if it's a list of lists (i.e., multiple rows of images), and calculates the width and height of the images in the array.
   #- It then resizes each image in the `imgArray` by the specified `scale` and converts grayscale images to color.
   #- Depending on whether `imgArray` is a list of lists or a single list, it horizontally or vertically stacks the images and returns the resulting stacked image.
   #- If `labels` are provided, it adds rectangles and text labels to the stacked image.

#2. `rectContour(contours):`
   #- This function takes a list of contours as input and returns a list of contours that are likely to represent rectangles or quadrilaterals.
   #- It filters the input contours based on their area, keeping only those with an area greater than 50.
   #- For the remaining contours, it approximates the shape using `cv2.approxPolyDP` and checks if the approximation has four corners (quadrilateral). It appends such contours to the `rectCon` list.
   #- Finally, it sorts the `rectCon` list based on contour area in descending order and returns the sorted list.

#3. `getCornerPoints(cont):`
   #- This function takes a single contour `cont` as input and returns the approximate corner points of the contour.
   #- It calculates the perimeter of the contour (`peri`) and approximates the contour using `cv2.approxPolyDP`. The result, `approx`, contains the corner points of the contour.
   #- It returns the corner points.

#4. `reorder(myPoints):`
   #- This function takes an array of corner points `myPoints` and reorders them such that they are in a specific order: top-left, top-right, bottom-left, and bottom-right.
   #- It reshapes the `myPoints` array and creates a new array, `myPointsNew`, to store the reordered points.
   #- It calculates the sum and difference of coordinates to identify the corner points' positions and assigns them to `myPointsNew`.
   #- It returns the reordered corner points.

#5. `splitBoxes(img):`
   #- This function takes an image `img` and splits it into a 5x5 grid of smaller boxes.
   #- It first vertically splits the image into 5 rows and then horizontally splits each row into 5 columns.
   #- It returns a list of the smaller box images.

#6. `showAnswers(img, myIndex, grading, ans, questions, choices):`
   #- This function is used to highlight the selected answers on an image of a multiple-choice questionnaire.
   #- It takes the following parameters:
   #  - `img`: The image of the questionnaire.
    # - `myIndex`: A list of indices representing the selected answers for each question.
    # - `grading`: A list indicating whether each answer is correct (1) or incorrect (0).
    # - `ans`: A list of the correct answers.
    # - `questions`: The number of questions in the questionnaire.
    # - `choices`: The number of answer choices for each question.
   #- It calculates the coordinates for highlighting the selected answers and correct answers (if needed) and adds circles with different colors to the image accordingly.
   #- It returns the modified image with highlighted answers.
