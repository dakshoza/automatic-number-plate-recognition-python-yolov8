from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort.sort import *
from util import get_car, read_license_plate
from datetime import datetime
import csv

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

def point_crossed_line(point, line_start, line_end, buffer=60):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    a=((y2-y1)/(x2-x1))
    a = round(a,2)
    b = -1
    c = -a*x1+y1
    c = round(c,2)

    distance = abs(a*px+b*py+c)/((a**2+b**2)**0.5)
    print(distance)
    
    # Check if the point is close enough to the line
    if distance > buffer:
        return False
    
    
    return True

results = {}
mot_tracker = Sort()

# load models
coco_model = YOLO('./yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')

# Get video properties for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

vehicles = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
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
roi_lines = draw_roi_lines(scaled_frame, scale)

print(roi_lines)

roi_lines = [((535, 6), (10, 459), 'North'), ((561, 7), (1905, 285), 'West'), ((10, 495), (1518, 1075), 'East'), ((1575, 1078), (1908, 378), 'South')]
# Open the CSV file before the main loop
csv_file = open('./test.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# Write the header row
csv_writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 
                     'license_plate_bbox_score', 'license_number', 'license_number_score',
                     'vehicle_type', 'recorded_time', 'entry_point', 'exit_point'])

# Dictionary to store entry and exit points for each tracked object
object_points = {}

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        vehicle_ids = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, vehicle_id = detection
            if int(vehicle_id) in vehicles.keys():
                detections_.append([x1, y1, x2, y2, score])
                vehicle_ids.append(int(vehicle_id))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # cv2.putText(frame, f"{vehicles[int(vehicle_id)]}", (int(x1), int(y1) - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]

        for (x1, y1, x2, y2, score), track, vehicle_id in zip(detections_, track_ids, vehicle_ids):
            track_id = int(track[4])

            cv2.putText(frame, f"{track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Initialize entry and exit points as None if this is a new tracked object
            if track_id not in object_points:
                object_points[track_id] = {'entry': None, 'exit': None}

            # Check for ROI crossings
            center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            print(center_point)
            print(x1, y1, x2, y2)

            for line_start, line_end, direction in roi_lines:
                print(direction)
                if point_crossed_line(center_point, line_start, line_end):
                    if object_points[track_id]['entry'] is None:
                        object_points[track_id]['entry'] = direction
                    else:
                        object_points[track_id]['exit'] = direction

            # Initialize with null values
            license_plate_bbox = None
            license_plate_text = None
            license_plate_score = None
            license_plate_text_score = None

            # Check if there's a license plate detection that fits within this vehicle's bounding box
            for license_plate in license_plates.boxes.data.tolist():
                lp_x1, lp_y1, lp_x2, lp_y2, lp_score, _ = license_plate

                if (lp_x1 > x1 and lp_y1 > y1 and lp_x2 < x2 and lp_y2 < y2):
                    license_plate_bbox = [lp_x1, lp_y1, lp_x2, lp_y2]
                    license_plate_score = lp_score

                    # crop license plate
                    license_plate_crop = frame[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2), :]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    # Draw bounding box for license plate
                    cv2.rectangle(frame, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (0, 0, 255), 2)
                    if license_plate_text:
                        cv2.putText(frame, license_plate_text, (int(lp_x1), int(lp_y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    break  # Assuming one license plate per vehicle

            # Always add the vehicle to results, whether it has a license plate or not
            vehicle_type = vehicles.get(vehicle_id, 'unknown')
            
            entry_point = object_points[track_id]['entry']
            exit_point = object_points[track_id]['exit']
            
            results[frame_nmr][track_id] = {
                'car': {'bbox': [x1, y1, x2, y2]},
                'license_plate': {
                    'bbox': license_plate_bbox,
                    'text': license_plate_text,
                    'bbox_score': license_plate_score,
                    'text_score': license_plate_text_score
                },
                'vehicle_type': vehicle_type,
                'recorded_time': current_time,
                'entry_point': entry_point,
                'exit_point': exit_point
            }

            # Write this vehicle's data to CSV
            csv_writer.writerow([
                frame_nmr,
                track_id,
                [x1, y1, x2, y2],
                license_plate_bbox,
                license_plate_score,
                license_plate_text,
                license_plate_text_score,
                vehicle_type,
                current_time,
                entry_point,
                exit_point
            ])

        # Draw ROI lines on the frame
        for line_start, line_end, direction in roi_lines:
            cv2.line(frame, line_start, line_end, (255, 0, 0), 2)
            cv2.putText(frame, direction, line_start, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Write the frame to output video
        out.write(frame)

        # Show the frame
        scaled_frame, _ = scale_image(frame)
        cv2.imshow('Frame', scaled_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop ends, close the CSV file
csv_file.close()

# Release the video capture and writer objects and close windows
cap.release()
out.release()
cv2.destroyAllWindows()