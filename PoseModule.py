# Import libraries
import cv2
import mediapipe as mp
import time
import math

# Define a class for pose detection
class poseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the poseDetector class with optional arguments:
        - mode: Whether to track continuous video feed or static images.
        - upBody: A flag for detecting upper body only (if applicable).
        - smooth: Whether to smooth pose landmarks over time.
        - detectionCon: Minimum confidence threshold for initial detection.
        - trackCon: Minimum confidence threshold for tracking landmarks.
        """
        self.mode = mode               # Set mode for static or video tracking
        self.upBody = upBody           # Flag for detecting only the upper body
        self.smooth = smooth           # Enable smoothing of pose landmarks
        self.detectionCon = detectionCon  # Minimum detection confidence
        self.trackCon = trackCon       # Minimum tracking confidence

        # Initialize MediaPipe utilities for drawing and pose estimation
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

        # Create a pose estimation model with the specified parameters
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=1,  # Complexity level 1 is used for the model
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        """
        Detects human pose landmarks in an image and optionally draws them.
        - img: The input image.
        - draw: Whether to draw the pose landmarks on the image.
        """
        # Convert the image to RGB since MediaPipe processes RGB images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image and get pose landmarks
        self.results = self.pose.process(imgRGB)
        
        # If landmarks are detected, optionally draw them
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img  # Return the image with landmarks drawn (if enabled)

    def findPosition(self, img, draw=True):
        """
        Extracts the x, y coordinates of detected landmarks and optionally draws them.
        - img: The input image.
        - draw: Whether to draw circles around the landmarks.
        """
        self.lmList = []  # Initialize an empty list to store landmark positions

        # If landmarks are detected
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape  # Get the dimensions of the image
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values
                self.lmList.append([id, cx, cy])  # Append the landmark ID and its position
                if draw:
                    # Draw a small circle on the detected landmark
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList  # Return the list of landmarks

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Calculates the angle between three points (landmarks) on the body.
        - img: The input image.
        - p1, p2, p3: Indices of the three landmarks to calculate the angle.
        - draw: Whether to draw lines and the angle value on the image.
        """
        # Get the x, y coordinates of the three points
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle between the three points using the arctangent of the slopes
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        
        # Ensure the angle is positive
        if angle < 0:
            angle += 360

        
        if draw:
            # Draw lines connecting the three points
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            
            # Draw circles at the points
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            
            # Display the calculated angle on the image
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle  # Return the calculated angle

def main():
    """
    Main function to capture video from webcam and perform pose detection in real time.
    """
    cap = cv2.VideoCapture(0)  # Capture video from the default webcam
    pTime = 0  # Variable to track the previous time for FPS calculation
    detector = poseDetector()  # Initialize the pose detector object

    while True:
        success, img = cap.read()  # Read a frame from the webcam
        img = detector.findPose(img)  # Detect the pose in the frame
        lmList = detector.findPosition(img, draw=False)  # Get landmark positions without drawing
        
        # If landmarks are detected, print the position of landmark 14 and draw a circle around it
        if len(lmList) != 0:
            print(lmList[14])  # Print the coordinates of landmark 14 (right elbow)
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)  # Draw a circle at landmark 14

        # Calculate and display the FPS (frames per second) on the image
        cTime = time.time()  # Current time
        fps = 1 / (cTime - pTime)  # FPS calculation
        pTime = cTime  # Update previous time for next frame
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)  # Display FPS

        # Show the image with the detected landmarks and FPS
        cv2.imshow("Image", img)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close any OpenCV windows

if __name__ == "__main__":
    main()  # Call the main function if this script is run directly
