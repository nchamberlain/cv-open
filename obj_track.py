'''
from https://courses.opencv.org/courses/course-v1:OpenCV+Bootcamp+CV0/courseware Lesson 12 - Object Tracking
Tracker Class in OpenCV Techniques
BOOSTING, MIL, KCF, CRST, TLD(Tends to recover from occulusions), MEDIANFLOW (Good for predictable slow motion),
  GOTURN (Deep Learning based, Most Accurate), MOSSE (Fastest)

IMPORTANT: Live video tracking NOT supported so all video frame updates are done offscreen

IMPORTANT: The support files goturn.caffemodel and goturn.prototxt must be downloaded separately from
"https://www.dropbox.com/s/ld535c8e0vueq6x/opencv_bootcamp_assets_NB11.zip?dl=1"   (approx 380MB)
These support files are ONLY needed for the GOTURN option so it is easiest to skip them unless you
REALLY REALLY want them. The zip file is from the OpenCV course and must be manually unzipped.
'''

import sys
import cv2
import matplotlib.pyplot as plt

video_input_file_name = "resources/video/race_car.mp4"


def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


def displayRectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy)
    plt.axis("off")


def drawText(frame, txt, location, color=(50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# Set up tracker
tracker_types = [
    "BOOSTING",
    "MIL",
    "KCF",
    "CSRT",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
]

# Change the index to change the tracker type
tracker_type = tracker_types[7] #MOSSE


if tracker_type == "BOOSTING":
    tracker = cv2.legacy.TrackerBoosting.create()
elif tracker_type == "MIL":
    tracker = cv2.legacy.TrackerMIL.create()
elif tracker_type == "KCF":
    tracker = cv2.TrackerKCF.create()
elif tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT.create()
elif tracker_type == "TLD":
    tracker = cv2.legacy.TrackerTLD.create()
elif tracker_type == "MEDIANFLOW":
    tracker = cv2.legacy.TrackerMedianFlow.create()
elif tracker_type == "GOTURN":
    tracker = cv2.TrackerGOTURN.create()
    #goturn.prototxt & goturn.caffemodel must be in same folder as obj_track.py
else:
    tracker = cv2.legacy.TrackerMOSSE.create()

# Read video
video = cv2.VideoCapture(video_input_file_name)
ok, frame = video.read()

# Exit if video not opened
if not video.isOpened():
    print("Could not open video")
    sys.exit()
else:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_output_file_name = "race_car-" + tracker_type + ".mp4"
print(video_output_file_name) #display the video out filename

#video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*"XVID"), 10, (width, height))
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)) #needed for Mac
# The 10 = frames per second, can set between 10 and 90, depending on speed of your computer

# Define a bounding box
bbox = (1300, 405, 160, 120)
# bbox = cv2.selectROI(frame, False) #Allows user to select Region of Interest (ROI)
# print(bbox)
displayRectangle(frame, bbox)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)\

while True:
    ok, frame = video.read()

    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        drawRectangle(frame, bbox)
    else:
        drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255))

    # Display Info
    drawText(frame, tracker_type + " Tracker", (80, 60))
    drawText(frame, "FPS : " + str(int(fps)), (80, 100))

    # Write frame to video
    video_out.write(frame)

video.release()
video_out.release()

'''
From a terminal window in the folder where the race_car-xxxx.mp4 file was saved, issue this command:

ffmpeg -y -i race_car-CSRT.mp4  -c:v libx264 "race_car_track_x264.mp4"  -hide_banner -loglevel error 

This command converts the .mp4 file originally saved into a mp4 format that is more universally accepted
'''