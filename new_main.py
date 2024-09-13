import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort.sort import *
from util import get_car, read_license_plate
from datetime import datetime
import csv
from collections import defaultdict

def scale_image(image, max_width=1280, max_height=720):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height)), scale
    return image, 1

def scale_coordinates(x, y, scale):
    return int(x / scale), int(y / scale)

def point_crossed_line(point, line_start, line_end, buffer=3):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate the distance from the point to the line
    numerator = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
    denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    distance = numerator / denominator
    
    # Check if the point is close enough to the line
    if distance > buffer:
        return False
    
    # Check if the point is within the line segment
    dot_product = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    line_length_squared = (x2 - x1)**2 + (y2 - y1)**2
    
    if dot_product < 0 or dot_product > line_length_squared:
        return False
    
    return True

results = {}

# load models
coco_model = YOLO('./models/best.pt')

# load video
cap = cv2.VideoCapture('./input.mp4')

# Get video properties for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_new.mp4', fourcc, fps, (frame_width, frame_height))

vehicles = {
    0: 'busrotation',
    1: 'bicyclerotation',
    2: 'autorotation',
    3: 'carrotation',
    4: 'temporotation',
    5: 'tractorrotation',
    6: 'two_wheelersrotation',
    7: 'vehicle_truckrotation'
}

# Function to draw ROI lines
def draw_roi_lines(frame, scale):
    lines = []
    directions = ['North', 'West', 'East', 'South']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for direction, color in zip(directions, colors):
        print(f"Draw {direction} line. Click two points.")
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                orig_x, orig_y = scale_coordinates(x, y, scale)
                points.append((orig_x, orig_y))
                cv2.circle(frame, (x, y), 5, color, -1)
                if len(points) == 2:
                    cv2.line(frame, 
                             (int(points[0][0]*scale), int(points[0][1]*scale)), 
                             (int(points[1][0]*scale), int(points[1][1]*scale)), 
                             color, 2)
                cv2.imshow('Frame', frame)

        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', mouse_callback)

        while len(points) < 2:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        lines.append((points[0], points[1], direction))
        cv2.setMouseCallback('Frame', lambda *args: None)

    cv2.destroyAllWindows()
    return lines

# Read the first frame and draw ROI lines
ret, first_frame = cap.read()
scaled_frame, scale = scale_image(first_frame)
roi_lines= draw_roi_lines(scaled_frame, scale)
final_list = {}


# Open the CSV file before the main loop
csv_file = open('new.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# # Write the header row
csv_writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 
                     'license_plate_bbox_score', 'license_number', 'license_number_score',
                     'vehicle_type', 'recorded_time', 'entry_point', 'exit_point'])

# Dictionary to store entry and exit points for each tracked object
object_points = {}

# read frames
frame_nmr = -1
ret = True
video_path = "./input.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# roi_lines = [((364, 213), (559, 52), 'North'), ((570, 52), (1057, 49), 'West'), ((1098, 70), (1498, 882), 'East'), ((88, 664), (1444, 972), 'South')]

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = coco_model.track(frame, persist=True)
        
        # detections = coco_model(frame)[0]
        detections_ = []
        vehicle_ids = {}
        vehicle_name = []
        final_detect = []
        for detection in results[0].summary():
                
                final_detect.append(detection)
                # if detection['name'] in vehicles.values():
                if detection['track_id'] in final_list.keys(): 
                    final_list[detection['track_id']] = detection
                else:
                    final_list[detection['track_id']] = detection

            # Variables if required

                # final_list.append(final_detect)
                # final_list = final_list.append(final_detect)
                # final_list = final_list
                # x1, y1, x2, y2 = detection['box']['x1'], detection['box']['y1'], detection['box']['x2'], detection['box']['y2']
                # score = detection['confidence']
                # vehicle_class = detection['class']


        print(final_list)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        for line_start, line_end, direction in roi_lines:
            cv2.line(annotated_frame, line_start, line_end, (255, 0, 0), 2)
            cv2.putText(annotated_frame, direction, line_start, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Show the frame

        # cv2.imshow("YOLOv8 Tracking", scaled_frame)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        out.write(annotated_frame)
        # out.write(scaled_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
# Release the video capture object and close the display window

# After the loop ends, close the CSV file
csv_file.close()
cap.release()
cv2.destroyAllWindows()


# Data to apend to CSV

    #         if track_id not in object_points:
    #             object_points[track_id] = {'entry': None, 'exit': None}






#         entry_point = object_points[track_id]['entry']
    #         exit_point = object_points[track_id]['exit']
            
    #         results[frame_nmr][track_id] = {
    #             'car': {'bbox': [x1, y1, x2, y2]},
    #             # 'license_plate': {
    #             #     'bbox': license_plate_bbox,
    #             #     'text': license_plate_text,
    #             #     'bbox_score': license_plate_score,
    #             #     'text_score': license_plate_text_score
    #             # },
    #             'vehicle_type': vehicle_type,
    #             'recorded_time': current_time,
    #             'entry_point': entry_point,
    #             'exit_point': exit_point
    #         }

    #         # Write this vehicle's data to CSV
    #         csv_writer.writerow([
    #             frame_nmr,
    #             track_id,
    #             [x1, y1, x2, y2],
    #             # license_plate_bbox,
    #             # license_plate_score,
    #             # license_plate_text,
    #             # license_plate_text_score,
    #             vehicle_type,
    #             current_time,
    #             entry_point,
    #             exit_point
    #         ])

