import csv

# Given dictionary
data_dict = {
    1: {'entry': 'East', 'exit': 'East'}, 
    2: {'entry': 'South', 'exit': 'South'}, 
    3: {'entry': None, 'exit': None}, 
    4: {'entry': None, 'exit': None}, 
    5: {'entry': None, 'exit': None}, 
    6: {'entry': None, 'exit': None}, 
    7: {'entry': 'North', 'exit': 'East'}, 
    8: {'entry': 'North', 'exit': 'East'}, 
    9: {'entry': 'North', 'exit': 'North'}, 
    10: {'entry': None, 'exit': None}, 
    11: {'entry': None, 'exit': None}, 
    12: {'entry': None, 'exit': None}, 
    13: {'entry': 'North', 'exit': 'North'}, 
    14: {'entry': 'West', 'exit': 'West'}, 
    15: {'entry': None, 'exit': None}
}

# Write the dictionary to a CSV file
csv_file = "output.csv"

# Define the headers
headers = ['track_id', 'entry', 'exit']

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(headers)
    
    # Write the data rows
    for track_id, points in data_dict.items():
        writer.writerow([track_id, points['entry'], points['exit']])

print(f"Data has been written to {csv_file}")
