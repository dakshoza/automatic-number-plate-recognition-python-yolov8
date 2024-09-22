from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from util import read_license_plate
from datetime import datetime
import csv
from improved_detection import improve_detections, draw_detections
from load_roi_lines import load_roi_lines

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
    
    numerator = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
    denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    distance = numerator / denominator
    
    if distance > buffer:
        return False
    
    dot_product = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    line_length_squared = (x2 - x1)**2 + (y2 - y1)**2
    
    if dot_product < 0 or dot_product > line_length_squared:
        return False
    
    return True

def main():
    results = {}
    mot_tracker = Sort()

    # Load models
    coco_model = YOLO('./yolov8n.pt')
    license_plate_detector = YOLO('./models/license_plate_detector.pt')

    # Load video
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

    # Load ROI lines
    roi_lines = load_roi_lines()

    # Open the CSV file before the main loop
    with open('./test.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 
                             'license_plate_bbox_score', 'license_number', 'license_number_score',
                             'vehicle_type', 'recorded_time', 'entry_point', 'exit_point'])

        # Dictionary to store entry and exit points for each tracked object
        object_points = {}

        # Read frames
        frame_nmr = -1
        while True:
            frame_nmr += 1
            ret, frame = cap.read()
            if not ret:
                break

            results[frame_nmr] = {}

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Improved detection
            detection_results = improve_detections(frame, coco_model, license_plate_detector, vehicles, mot_tracker)

            # Process detection results
            for result in detection_results:
                vehicle = result['vehicle']
                license_plate = result['license_plate']
                track_id = vehicle['id']

                # add scale coords here?
                vehicle['bbox'] = [
                    scale_coordinates(vehicle['bbox'][0], vehicle['bbox'][1], 2),
                    scale_coordinates(vehicle['bbox'][2], vehicle['bbox'][3], 2)
                ]

                # Scale license plate bounding box coordinates if detected
                if license_plate:
                    license_plate['bbox'] = [
                        scale_coordinates(license_plate['bbox'][0], license_plate['bbox'][1], 2),
                        scale_coordinates(license_plate['bbox'][2], license_plate['bbox'][3], 2)
                    ]
                # Initialize entry and exit points as None if this is a new tracked object
                if track_id not in object_points:
                    object_points[track_id] = {'entry': None, 'exit': None}

                # Check for ROI crossings
                center_point = ((vehicle['bbox'][0] + vehicle['bbox'][2]) / 2, 
                                (vehicle['bbox'][1] + vehicle['bbox'][3]) / 2)
                for line_start, line_end, direction in roi_lines:
                    if point_crossed_line(center_point, line_start, line_end):
                        if object_points[track_id]['entry'] is None:
                            object_points[track_id]['entry'] = direction
                        else:
                            object_points[track_id]['exit'] = direction

                results[frame_nmr][track_id] = {
                    'car': {'bbox': vehicle['bbox']},
                    'license_plate': {
                        'bbox': license_plate['bbox'] if license_plate else None,
                        'text': None,
                        'bbox_score': license_plate['score'] if license_plate else None,
                        'text_score': None
                    },
                    'vehicle_type': vehicle['type'],
                    'recorded_time': current_time,
                    'entry_point': object_points[track_id]['entry'],
                    'exit_point': object_points[track_id]['exit']
                }

                # Process license plate if detected
                if license_plate:
                    license_plate_crop = frame[int(license_plate['bbox'][1]):int(license_plate['bbox'][3]), 
                                               int(license_plate['bbox'][0]):int(license_plate['bbox'][2])]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    results[frame_nmr][track_id]['license_plate']['text'] = license_plate_text
                    results[frame_nmr][track_id]['license_plate']['text_score'] = license_plate_text_score

                # Write this vehicle's data to CSV
                csv_writer.writerow([
                    frame_nmr,
                    track_id,
                    vehicle['bbox'],
                    license_plate['bbox'] if license_plate else None,
                    license_plate['score'] if license_plate else None,
                    license_plate_text if license_plate else None,
                    license_plate_text_score if license_plate else None,
                    vehicle['type'],
                    current_time,
                    object_points[track_id]['entry'],
                    object_points[track_id]['exit']
                ])

            # Draw detections and ROI lines on the frame
            frame = draw_detections(frame, detection_results)
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

    # Release the video capture and writer objects and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()