import csv
import random
# Input text file path
input_file = "text2.txt"

# Output CSV file path
output_file = "output.csv"

# Create and open the CSV file for writing
with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    # Initialize the CSV writer
    csv_writer = csv.writer(csvfile)
    
    # Write the header row
    csv_writer.writerow(["label", "text"])
    
    # Open and read the input text file
    
    with open(input_file, mode="r", encoding="utf-8") as txtfile:
        lines = txtfile.readlines()
        random.shuffle(lines)
        for line in lines:
            # Strip any leading/trailing whitespace from the line
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Split the line into label (first word) and text (remaining sentence)
            parts = line.split(maxsplit=1)
            label = parts[0]  # First word as label
            text = parts[1] if len(parts) > 1 else ""  # Remaining sentence as text
            
            # Write the label and text to the CSV
            csv_writer.writerow([label, text])

print(f"CSV file '{output_file}' has been created successfully.")
