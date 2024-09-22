import cv2
import numpy as np
from ultralytics import YOLO

def improve_detections(frame, coco_model, license_plate_detector, vehicles, mot_tracker):
    # Detect vehicles
    vehicle_results = coco_model(frame)[0]
    detections = []
    vehicle_ids = []
    
    for detection in vehicle_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles.keys():
            detections.append([x1, y1, x2, y2, score])
            vehicle_ids.append(int(class_id))
    
    # Track vehicles
    track_ids = mot_tracker.update(np.array(detections))
    
    # Detect license plates
    license_plate_results = license_plate_detector(frame)[0]
    
    results = []
    for (x1, y1, x2, y2, score), track, vehicle_id in zip(detections, track_ids, vehicle_ids):
        track_id = int(track[4])
        
        # Find the best matching license plate
        best_plate = None
        best_iou = 0
        for license_plate in license_plate_results.boxes.data.tolist():
            lp_x1, lp_y1, lp_x2, lp_y2, lp_score, _ = license_plate
            
            # Calculate IoU between vehicle and license plate
            iou = calculate_iou([x1, y1, x2, y2], [lp_x1, lp_y1, lp_x2, lp_y2])
            
            if iou > best_iou and lp_x1 > x1 and lp_y1 > y1 and lp_x2 < x2 and lp_y2 < y2:
                best_iou = iou
                best_plate = license_plate
        
        result = {
            'vehicle': {
                'bbox': [x1, y1, x2, y2],
                'score': score,
                'id': track_id,
                'type': vehicles.get(vehicle_id, 'unknown')
            },
            'license_plate': None
        }
        
        if best_plate:
            lp_x1, lp_y1, lp_x2, lp_y2, lp_score, _ = best_plate
            result['license_plate'] = {
                'bbox': [lp_x1, lp_y1, lp_x2, lp_y2],
                'score': lp_score
            }
        
        results.append(result)
    
    return results

def calculate_iou(box1, box2):
    # Calculate intersection over union
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / (area1 + area2 - intersection + 1e-6)
    return iou

def draw_detections(frame, results):
    for result in results:
        vehicle = result['vehicle']
        license_plate = result['license_plate']
        
        # Draw vehicle bounding box
        cv2.rectangle(frame, 
                      (int(vehicle['bbox'][0]), int(vehicle['bbox'][1])), 
                      (int(vehicle['bbox'][2]), int(vehicle['bbox'][3])), 
                      (0, 255, 0), 2)
        cv2.putText(frame, f"{vehicle['type']} {vehicle['id']}", 
                    (int(vehicle['bbox'][0]), int(vehicle['bbox'][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw license plate bounding box if detected
        if license_plate:
            cv2.rectangle(frame, 
                          (int(license_plate['bbox'][0]), int(license_plate['bbox'][1])), 
                          (int(license_plate['bbox'][2]), int(license_plate['bbox'][3])), 
                          (0, 0, 255), 2)
    
    return frame