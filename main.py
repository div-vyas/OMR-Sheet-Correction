import cv2#importing computer vision module
import numpy as np#image Representation #Array Operations #Array Slicing and Region of Interest (ROI) Selection #Array Manipulation #Array Stacking and Splitting #Mathematical and Statistical Analysis # Broadcasting
import utlis #Custom module created to stack all images together in an array so that the pipleline of the project can be observed accordingly

########################
#PARAMETERS
path="1.jpeg"#PATH OF IMAGE AS STORED IN THE FILE
widthImg=700#WIDTH OF IMAGE USED FOR RESIZING THE ACTUAL IMAGE
heightImg=700#HEIGHT OF IMAGE USED FOR RESIZING THE ACTUAL IMAGE
questions=5#NUMBER OF QUESTIONS DEFINED
choices=5#NUMBER OF CHOICES DEFINED
ans=[1,2,0,1,4]#ANSWER KEY
webcamFeed = True#The line of code webcamFeed = True suggests that a webcam feed is being utilized in your project. Setting webcamFeed to True typically indicates that you want to access and process live video from a connected webcam.
cameraNo=0#default webcam index ELSE 1 for external cam
#######################

cap=cv2.VideoCapture(cameraNo)#This line creates a video capture object named cap.The cameraNo variable represents the camera index or number. By passing the camera index to cv2.VideoCapture(), you are initializing the video capture object to read frames from the specified camera.
cap.set(10,150)#used to modify various properties of the video capture object.In this case the property ID 10 corresponds to cv2.CAP_PROP_BRIGHTNESS, and the value 150 is being set as the brightness level.

while True:
    if webcamFeed:success,img=cap.read()#This condition checks if webcamFeed is True. If it is, it means the webcam feed is being used. cap.read() retrieves the next frame from the video capture object cap, and the resulting frame is stored in the img variable.
    else:img=cv2.imread (path)#if webcamFeed is False, it means an image file is being used instead of the webcam feed. In this case, cv2.imread(path) reads an image file specified by the path variable, and the resulting image is stored in the img variable.


    #PROCESSING
    img=cv2.resize(img,(widthImg,heightImg))#resizing the orignal image.Resized Width and height parameters have already been defined.
    imgContours=img.copy()#The line of code imgContours = img.copy() creates a copy of the original image img and assigns it to a new variable imgContours.
    imgFinal = img.copy()#The line of code imgFinal = img.copy() creates a copy of the original image img and assigns it to a new variable imgFinal.
    imgBiggestcontours = img.copy()#The line of code imgBiggestcontours = img.copy() creates a copy of the original image img and assigns it to a new variable imgBiggestcontours.
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#Converting orignal image to gray scale.This process is termed as Gray scale conversion.
    imgBlur=cv2.GaussianBlur((imgGray),(5,5),1)#done to ensure:simplicity,dimesnsionality reduction,illumination invariance,noise reduction,compatibility #(5,5) is the size of gaussian kernel i.e. blurring operation will consider a neighbourhood of 5x5 pixels around each pixels # 1 indiscates the standard deviation of gaussian distribution used for blurring operation 
    imgCanny=cv2.Canny(imgBlur,10,50)#done to ensure edge detection #(10,50) are threshold values. ANy gradient magnitude above or below these threshold values will be considered as non edged or strongly edged pixels.

    try:
        #FINDING ALL CONTOURS
        contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#his line applies the cv2.findContours() function to detect contours in the image imgCanny. The imgCanny is assumed to be a binary image obtained from previous image processing steps, such as Canny edge detection. The function returns a list of contours (contours) and a hierarchy representation (hierarchy). The cv2.RETR_EXTERNAL flag specifies that only the external contours are retrieved, while cv2.CHAIN_APPROX_NONE indicates that all contour points are stored.
        cv2.drawContours(imgContours,contours,-1,(0,255,0),10)#This line draws the detected contours on the imgContours image using the cv2.drawContours() function. The imgContours image is assumed to be a copy of the original image where contours will be visualized. The contours variable contains the list of contours obtained in the previous step. The -1 parameter indicates that all contours should be drawn. The (0, 255, 0) argument specifies the color of the contour (here, green in the BGR format), and 10 is the thickness of the contour lines.

        rectCon=utlis.rectContour(contours)#This line calls a function rectContour from a custom module utlis. This function processes the list of contours (contours) and returns the contours representing the bounding rectangles of each contour. The rectCon variable stores the resulting contours.
        biggestContour=utlis.getCornerPoints(rectCon[0])# This line calls another function getCornerPoints from the utlis module to extract the corner points of the largest contour (rectCon[0]). The biggestContour variable holds the corner points of the largest contour.
        gradePoints = utlis.getCornerPoints(rectCon [1])#Similarly, this line retrieves the corner points of the second largest contour (rectCon[1]) using the getCornerPoints function. The gradePoints variable stores the corner points.
        #print(biggestContour)


        if biggestContour.size != 0 and gradePoints.size != 0:#The provided code snippet includes an if condition that checks if the biggestContour and gradePoints arrays have non-zero sizes
            cv2.drawContours(imgBiggestcontours,biggestContour,-1,(0,255,0),20)#If the biggestContour array has a non-zero size, this line draws the contour on the imgBiggestcontours image using the cv2.drawContours() function. The biggestContour variable contains the corner points of the largest contour, and -1 indicates that all contour segments should be drawn. The (0, 255, 0) argument specifies the color of the contour (green in the BGR format), and 20 is the thickness of the contour lines.
            cv2.drawContours(imgBiggestcontours, gradePoints,-1, (255, 0, 0), 20)#Similarly, if the gradePoints array has a non-zero size, this line draws the contour on the imgBiggestcontours image. The gradePoints variable contains the corner points of the second largest contour. The (255, 0, 0) argument specifies the color of the contour (blue in the BGR format), and 20 is the thickness of the contour lines.

            biggestContour=utlis.reorder(biggestContour)#This line calls the reorder function from the utlis module and passes the biggestContour array as an argument. It is likely that the reorder function is designed to reorder the corner points of a contour to a specific order or arrangement. By calling this function, the corner points of the biggestContour are reordered according to the predefined logic implemented in the reorder function.
            gradePoints=utlis.reorder(gradePoints)# Similarly, this line calls the reorder function from the utlis module and passes the gradePoints array as an argument. It reorders the corner points of the gradePoints contour array using the reorder function.

            pt1 = np.float32(biggestContour)# This line converts the biggestContour array, which contains the corner points of the largest contour, to a NumPy array of type float32. The np.float32() function is used to ensure that the array elements are in the floating-point format required for the perspective transformation.
            pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])#Here, a new NumPy array pt2 is created. It contains the desired destination points for the perspective transformation. The four corner points of the destination image are specified as [[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]]. These points define the shape and size of the output warped image.
            matrix = cv2.getPerspectiveTransform(pt1,pt2)# The cv2.getPerspectiveTransform() function computes the perspective transformation matrix based on the source points (pt1) and the destination points (pt2). This matrix represents the mapping between the source and destination image coordinates.
            imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))#The cv2.warpPerspective() function applies the perspective transformation to the input image img using the transformation matrix matrix. The resulting warped image is stored in the imgWarpColored variable. The output image has the specified dimensions (widthImg, heightImg).

            ptG1 = np.float32(gradePoints)#This line converts the gradePoints array, which contains the corner points of the second largest contour, to a NumPy array of type float32. The np.float32() function ensures that the array elements are in the floating-point format required for the perspective transformation.
            ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])#Here, a new NumPy array ptG2 is created. It contains the desired destination points for the perspective transformation specific to the grade area. The four corner points of the destination grade area are specified as [[0, 0], [325, 0], [0, 150], [325, 150]]. These points define the shape and size of the output warped grade area.
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)# The cv2.getPerspectiveTransform() function computes the perspective transformation matrix based on the source points (ptG1) and the destination points (ptG2) specific to the grade area. This matrix represents the mapping between the source and destination image coordinates for the grade area.
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))#The cv2.warpPerspective() function applies the perspective transformation to the input image img using the transformation matrix matrixG specific to the grade area. The resulting warped image is stored in the imgGradeDisplay variable. The output image has the specified dimensions (325, 150).
            #cv2.imshow("Grade",imgGradeDisplay)

            imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)#This line converts the color image imgWarpColored to grayscale using the cv2.cvtColor() function. The cv2.COLOR_BGR2GRAY flag is used to specify the color conversion from BGR (common for color images) to grayscale. The resulting grayscale image is stored in the imgWarpGray variable.
            imgThresh=cv2.threshold(imgWarpGray,150,255,cv2.THRESH_BINARY_INV)[1]#This line applies a thresholding operation to the grayscale image imgWarpGray using the cv2.threshold() function. The thresholding operation separates the image into foreground and background based on a specified threshold value.

            boxes = utlis.splitBoxes(imgThresh)# This line calls the splitBoxes function from the utlis module and passes the imgThresh image as an argument. The splitBoxes function is likely implemented to divide the thresholded image into individual boxes or regions of interest.
            #cv2.imshow("Test",boxes[2])
            #print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))

            #GETTING NON ZERO PIXEL VALUES OF EACH BOX
            myPixelVal=np.zeros((questions,choices))#This line creates a NumPy array myPixelVal filled with zeros. The shape of the array is specified as (questions, choices), where questions represents the number of rows and choices represents the number of columns. Each element in the array is initialized to zero. This array is likely intended to store pixel values or information related to questions and choices in your project.
            countC = 0#his line sets the variable countC to zero. It is likely used as a counter or index to keep track of the current column while iterating over the columns of myPixelVal
            countR = 0#This line sets the variable countR to zero. It is likely used as a counter or index to keep track of the current row while iterating over the rows of myPixelVal

            for image in boxes:#This line starts a loop that iterates over each image in the boxes list. It assumes that boxes contains a collection of images or regions of interest.
                totalPixels=cv2.countNonZero(image)#This line calculates the total number of non-zero (foreground) pixels in the current image using the cv2.countNonZero() function. The image represents an individual box or region of interest extracted from the thresholded image.
                myPixelVal[countR][countC]=totalPixels#This line assigns the totalPixels count to the corresponding position in the myPixelVal array at the current row countR and column countC. This stores the pixel count for the specific question and choice combination represented by the indices.
                countC+=1#his line increments the countC variable to move to the next column in the myPixelVal array.
                if(countC==choices):countR +=1 ;countC=0#This condition checks if the countC variable has reached the number of choices. If so, it means that all choices for the current question have been processed, and it increments the countR variable to move to the next row and resets the countC variable to zero.
            #print(myPixelVal)

            #FINDING INDEX VALUES OF THE MARKINGS
            myIndex = []#This line initializes an empty list named myIndex that will store the indices of the maximum values in each row of the myPixelVal array.
            for x in range (0,questions):#This line starts a loop that iterates over the range of values from 0 to questions (exclusive). It assumes that questions represents the number of rows in the myPixelVal array.
                arr=myPixelVal[x]#This line retrieves the x-th row of the myPixelVal array and assigns it to the arr variable.
                #print("arr",arr)
                myIndexVal = np.where(arr==np.amax(arr))#This line uses the np.where() function to find the indices where the maximum value occurs in the arr array. The np.amax() function is used to calculate the maximum value in the arr array. The resulting indices are assigned to the myIndexVal variable.
                #print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])#his line appends the first element of the myIndexVal array (which represents the index of the maximum value) to the myIndex list.
            #print(myIndex)

            #GRADING
            grading=[]#This line initializes an empty list named grading that will store the grading results for each question
            for x in range (0,questions):# This line starts a loop that iterates over the range of values from 0 to questions (exclusive). It assumes that questions represents the number of questions or the length of the ans and myIndex lists.
                if ans[x]==myIndex[x]:#This line checks if the x-th element of the ans list is equal to the x-th element of the myIndex list, indicating a correct answer for the corresponding question.
                    grading.append(1)#If the condition in the previous step is true, meaning the answer is correct, this line appends 1 to the grading list, indicating a correct grade for that question.
                else:grading.append(0)#If the condition in step 3 is false, meaning the answer is incorrect, this line appends 0 to the grading list, indicating an incorrect grade for that question.
            #print(grading)

            score= (sum(grading)/questions)*100 #FINAL_SCORE
            print(score)

            #DISPLAYING ANSWERS
            imgResult = imgWarpColored.copy()# This line creates a copy of the imgWarpColored image and assigns it to the imgResult variable.
            imgResult = utlis.showAnswers(imgResult,myIndex,grading,ans,questions,choices)#This line calls the showAnswers function from the utlis module to display the answers and grading information on the imgResult image. It takes imgResult as the input image and the myIndex, grading, ans, questions, and choices variables as arguments.
            imRawDrawing = np.zeros_like(imgWarpColored)#his line creates a black image (imRawDrawing) with the same dimensions as the imgWarpColored image.
            imRawDrawing = utlis.showAnswers(imRawDrawing, myIndex, grading, ans, questions, choices)#This line calls the showAnswers function to display the answers and grading information on the imRawDrawing image.
            invmatrix = cv2.getPerspectiveTransform(pt2, pt1)#This line computes the inverse perspective transformation matrix (invmatrix) by using the destination points (pt2) as source points and the source points (pt1) as destination points. It is used to revert the perspective transformation later.
            imgInvWarp = cv2.warpPerspective(imRawDrawing, invmatrix, (widthImg, heightImg))#This line applies the inverse perspective transformation to the imRawDrawing image using the invmatrix matrix. The resulting image is stored in imgInvWarp, which will bring the drawing back to the original perspective.

            imgRawGrade=np.zeros_like(imgGradeDisplay)#This line creates a black image (imgRawGrade) with the same dimensions as the imgGradeDisplay image.
            cv2.putText(imgRawGrade,str(int(score))+"%", (60,100), cv2.FONT_HERSHEY_COMPLEX,3,(0,255 ,255),3)#This line adds the text representing the score to the imgRawGrade image using the cv2.putText() function. The score is converted to an integer and displayed at position (60, 100) with a specific font, size, and color.
            #cv2.imshow("Grade",imgRawGrade)
            invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)#This line adds the text representing the score to the imgRawGrade image using the cv2.putText() function. The score is converted to an integer and displayed at position (60, 100) with a specific font, size, and color.
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))# This line applies the inverse perspective transformation to the imgRawGrade image using the invMatrixG matrix. The resulting image is stored in imgInvGradeDisplay.

            imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)#This line combines the imgFinal image and the imgInvWarp image using the cv2.addWeighted() function. The two images are added together with equal weights and no gamma correction, and the result is stored in imgFinal.
            imgFinal = cv2.addWeighted(imgFinal, 1,imgInvGradeDisplay, 1, 0)#This line further combines the imgFinal image with the imgInvGradeDisplay image using cv2.addWeighted(). The two images are added together with equal weights and no gamma correction, and the result is stored in imgFinal.





        imgBlank=np.zeros_like(img)#This line creates a black image (imgBlank) with the same dimensions as the img image. The np.zeros_like() function creates an array of zeros with the same shape and data type as the specified input array (img). In this case, it creates an array filled with zeros that has the same shape (height, width, and number of channels) as the img image.the purpose of creating this black image is to have a blank canvas with the same dimensions as the original image (img)
        imageArray = ([img,imgGray,imgBlur,imgCanny],
                      [imgContours,imgBiggestcontours,imgWarpColored,imgThresh],
                      [imgResult,imRawDrawing,imgInvWarp, imgFinal])
    except:
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgBlank,imgBlank, imgBlank, imgBlank],
                      [imgBlank,imgBlank, imgBlank, imgBlank])


    lables = [["Original","Gray","Blur","Canny"],
               ["Contours","Biggest Con","Warp","Threshold"],
               ["Result","Raw Drawing","Inv Warp","Final"]]
    imgStacked=utlis.stackImages(imageArray,0.3)# This line calls the stackImages function from the utlis module and passes imageArray and 0.3 as arguments. The imageArray is assumed to be a list or array containing the images to be stacked together, and 0.3 represents the scale factor applied to the images.

    cv2.imshow("Final Result",imgFinal)
    cv2.imshow("stacked images",imgStacked)
    if cv2.waitKey(1) & 0xFF==ord('s'):#This line checks for a keyboard event using the cv2.waitKey() function with a delay of 1 millisecond. The expression & 0xFF is used to ensure compatibility with both 32-bit and 64-bit systems. The condition == ord('s') checks if the pressed key corresponds to the letter 's'.
        cv2.imwrite("FinalResult.jpg",imgFinal)#If the condition in the previous line evaluates to true, this line saves the imgFinal image as a file named "FinalResult.jpg" using the cv2.imwrite() function. The first argument specifies the filename, and the second argument is the image to be saved.
        cv2.waitKey(300)#This line adds a delay of 300 milliseconds after saving the image. It allows the image to remain visible for a short duration before the program exits.
