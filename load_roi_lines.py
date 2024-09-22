import json

def load_roi_lines(filename='roi_lines.json'):
    """
    Load stored coordinates for ROI lines from a JSON file.

    Args:
        filename (str): Name of the JSON file containing ROI line coordinates.

    Returns:
        list: List of tuples, each containing start point, end point, and direction of an ROI line.
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        roi_lines = []
        for line in data:
            start = tuple(line['start'])
            end = tuple(line['end'])
            direction = line['direction']
            roi_lines.append((start, end, direction))
        
        return roi_lines
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please make sure the file exists.")
        return []
    except json.JSONDecodeError:
        print(f"Error: {filename} is not a valid JSON file.")
        return []
    except KeyError as e:
        print(f"Error: Missing key in JSON file: {e}")
        return []