#Saad Ahmed
#Smile Detection Project 
#introduction to Python & Machine Learning Project


#Using OpenCV Library I created a program that access the users Webcam
#which is then used to find individual frames every milisecond
#these frames are then examined using the 2 XML files for face and smile
#first the frame is coonfirmed to contain a face, then from there it is 
#used to find the a possible smile is a smile exist in the face: the message
#"Awesome Smile! You Should Smile More :)" is displayed to the user 


import cv2 

#Face Finder
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#Get Webcam footage
webcam = cv2.VideoCapture(0)

while True:
    # finds the current frame from the webcam stream    
    successful_frame_read, frame = webcam.read()

    #quits loop if frame read is not sucessful
    if not successful_frame_read:
        break

    #changes frame to greyscale, this greatly increases accuracy of the detection due to less color channels in greyscale
    frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face first 
    faces = face_detector.detectMultiScale(frame_greyscale)
    

    #runs smile detection
    for (x, y, w, h) in faces: 
        #draws rectangle
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,200,50), 5)

        #get the subframe using (using numpy & N-dimensional array slicing)
        the_face= frame[y:y+w, x:x+w]

        face_greyscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_greyscale, scaleFactor = 1.7, minNeighbors= 20)

        #for (x2, y2, w2, h2) in smiles:
            #cv2.rectangle(the_face, (x2 ,y2 ), (x2 +w2 , y2 +h2 ), (50,50,100), 5)

        if len(smiles) >0:
            cv2.putText(frame,"Awesome Smile! You Should Smile More :)", (x-400, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

     
    #displays the frame
    cv2.imshow('SmileMore', frame)

    # moves to the next frame after 1 milisecond, making the frames into a live video
    cv2.waitKey(1) 

#Cleaning
webcam.release()
cv2.destroyAllWindows()
