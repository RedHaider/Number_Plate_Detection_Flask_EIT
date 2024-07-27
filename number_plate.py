import cv2
import os
import time

# Load the Haar Cascade for plate detection
harcascade = "models/haarcascade_russian_plate_number.xml"

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

# Set frame width and height
cap.set(3, 700)  # width 
cap.set(4, 500)  # height

# Define minimum area for detected plates and initialize counter
min_area = 600
count = 0

# Ensure the output directory exists
output_dir = "plates"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Time interval for saving images (1/4 seconds for 4 images per second)
save_interval = 0.25
last_save_time = time.time()

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    
    # Convert to grayscale for better detection accuracy
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the Haar Cascade classifier
    plate_cascade = cv2.CascadeClassifier(harcascade)
    
    # Detect plates in the frame
    plates = plate_cascade.detectMultiScale(img_grey, 1.1, 4)
    
    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            # Draw rectangle around the plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 240, 150), 3)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            
            # Extract the region of interest (ROI)
            img_roi = img[y: y + h, x: x + w]
            cv2.imshow("ROI", img_roi)
            
            # Save the detected plate image automatically at the specified interval
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                filename = os.path.join(output_dir, f"scanned_img_{count}.jpg")
                cv2.imwrite(filename, img_roi)
                print(f"Plate saved as {filename}")
                last_save_time = current_time
                count += 1

    # Display the resulting frame
    cv2.imshow("Result", img)
    
    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
