# Import libraries
import cv2
import numpy as np
import time
import PoseModule as pm  # Import a custom module for pose detection (assumed to be PoseModule)

# Capture video from the webcam (0 refers to the default webcam)
cap = cv2.VideoCapture(0)

# Initialize the pose detector object from the custom PoseModule
detector = pm.poseDetector()

# Variables to count the number of curls and to track direction (up/down)
count = 0
dir = 0  # 0: moving down, 1: moving up

# Variable to store the previous time for calculating FPS (frames per second)
pTime = 0

# Start an infinite loop to continuously capture and process frames from the webcam
while True:
    # Capture a frame from the video feed
    success, img = cap.read()

    # Resize the image to a fixed width and height (1280x720 pixels)
    img = cv2.resize(img, (1280, 720))

    
    # img = cv2.imread("AiTrainer/test.jpg")

    # Use the pose detector to find the pose landmarks in the image, without drawing them
    img = detector.findPose(img, False)

    # Get the list of keypoint positions from the detected pose
    lmList = detector.findPosition(img, False)

    # If landmarks are detected (list is not empty)
    if len(lmList) != 0:
        # Find the angle of the right arm (between shoulder, elbow, and wrist)
        angle = detector.findAngle(img, 12, 14, 16)

        # Alternatively, find the angle of the left arm (commented out here)
        # angle = detector.findAngle(img, 11, 13, 15, False)

        # Convert the angle to a percentage (from 0 to 100) for easier tracking
        per = np.interp(angle, (210, 310), (0, 100))

        # Convert the angle to a value for a progress bar display (range: 100 to 650)
        bar = np.interp(angle, (220, 310), (650, 100))

        # Initialize the color for the bar display (default is purple)
        color = (255, 0, 255)

        # If the percentage is 100 (i.e., arm fully curled up)
        if per == 100:
            color = (0, 255, 0)  # Change the bar color to green
            if dir == 0:  # If the previous direction was down (curling up now)
                count += 0.5  # Increase the count by 0.5
                dir = 1  # Change direction to up

        # If the percentage is 0 (i.e., arm fully extended down)
        if per == 0:
            color = (0, 255, 0)  # Change the bar color to green
            if dir == 1:  # If the previous direction was up (curling down now)
                count += 0.5  # Increase the count by 0.5
                dir = 0  # Change direction to down

        # Print the current curl count in the console
        print(count)

        # Draw the progress bar on the image
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)  # Outer rectangle
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)  # Filled bar based on curl progress
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)  # Display percentage

        # Draw the curl count box on the image
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)  # Filled green rectangle
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)  # Display the count

    # Calculate the current time and frame rate (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)  # FPS is the inverse of time taken to process a frame
    pTime = cTime  # Update previous time for the next iteration

    # Display the FPS on the image
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    cv2.imshow("Image", img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
