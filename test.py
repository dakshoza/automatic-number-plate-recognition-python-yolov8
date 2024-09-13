import numpy as np

def point_crossed_line(point, line_start, line_end, buffer=5):
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


point = (1,1)
line_start = (0,0)
line_end = (8,0)

print(point_crossed_line(point, line_start, line_end, buffer=0))