# SmartSortingSystem
Real time cans and bottles detection and tracking

![](ezgif.com-video-to-gif.gif)


I kick-started the project by collecting and annotating an Object Detection with bounding boxes Dataset for crushed cans and bottles. Next, I implemented it into the SSD MobileNetV2 training model. Then, I converted the output TensorFlow model into a TFlite model.

To test the capabilities of the trained model, I incorporated it into Python scripts, allowing it to handle various input cases, such as video, image, and live webcam feeds. On Windows, I used the Interpreter module from the TensorFlow library, while on the Raspberry Pi, I relied on the TFLite runtime for model inference.

For the Tracking part, I implemented it using the centroid tracking method. Then, I incorporated OpenCV to visualize the model's output.

To estimate the position of detected objects relative to the image frame in millimeters, I employed ArUco markers. By obtaining a scaling factor to convert pixels to millimeters. I tested the accuracy of the object positions by obtaining the transformation between the image frame and a two-link manipulator base frame. The results demonstrated a successful match, however, I don't think this method is practical due to the camera distortion.

a suggested solution was to perform camera calibration. Using the resulting camera matrix and distortion coefficient to find the pose of a marker placed at the robot's base frame. While my initial results weren't satisfactory, I am actively seeking solutions and open to any insights and assistance on this matter.

In addition to detecting and tracking objects, I added a part to the script that prints the position, track ID, and class of any object at the moment of its first detection.
