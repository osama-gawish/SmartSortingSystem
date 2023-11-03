import cv2
import numpy as np
import importlib.util
import math
import time
from threading import Thread


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


# pkg = importlib.util.find_spec('tflite_runtime')
# if pkg:
from tflite_runtime.interpreter import Interpreter

# else:
#from tensorflow.lite.python.interpreter import Interpreter


VIDEO_PATH = "bottle2.mp4"
PATH_TO_CKPT = "detect.tflite"
PATH_TO_LABELS = "labelmap.txt"
min_conf_threshold = 0.5
resW, resH = 640, 480
imW, imH = int(resW), int(resH)


# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(640,480),framerate=30).start()
time.sleep(1)

# Initialize count
count = 0
center_points_prev_frame = []
tracking_objects = {}
track_id = 1
passed = 0

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()
    
    
    count += 1
    # if not ret:
    #   print('Reached the end of the video!')
    #   break

    center_points_cur_frame = []

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if scores[i] > min_conf_threshold:

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cx = int((xmax + xmin)/2)
            cy = int((ymax + ymin)/2)
            center_points_cur_frame.append((cx, cy, int(classes[i])))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 0), 4)
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # Draw label text

    ########## Tracking
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1

                    
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, (pt[0], pt[1]), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
    center_points_prev_frame = center_points_cur_frame.copy()
    
    #################
    
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    cv2.imshow('Object detector', frame)
    print(tracking_objects) #tracking_objects is a dictionary containning the information of each frame 
    # Convert tracking_objects to a numpy array where first element of each row is a tracking id,
    # second is X position, third is Y position, and forth is classID
    trk = np.array([(k,) + v for k, v in tracking_objects.items()])
    print(trk)

    
    traked = np.empty((0, 4), dtype=int)
    for row in trk:
        if row[0] > passed:
            passed+=1
            traked = np.vstack((traked, row))

    pick = np.empty((traked.shape[0], 3),dtype=int)
    place = np.empty((traked.shape[0], 3),dtype=int)
    for i in range(traked.shape[0]):
        # Replace the zeros with the Inverse Kinematics equations of the Delta robot
        th1 = 0
        th2 = 0
        th3 = 0
        pick[i] = [th1, th2, th3]
        if traked[i][3] == 0:
            place[i] = [20, 18, 25] # Assume a place position for bottles
            
        else:
            place[i] = [25, 17, 10] # Assume a place position for cans
            
    # print(pick)
    # print(place)
    
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    
    
# Clean up
cv2.destroyAllWindows()
videostream.stop()
