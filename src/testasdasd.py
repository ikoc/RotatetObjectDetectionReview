import re

line = "Average Precision  (AP) @[ IoU=0.50      | area=   all | angle=   -30 | maxDets=500 ] = 0.967"

# Define a regular expression pattern to match the AP value
ap_pattern = re.compile(r"= (\d+\.\d+)")

# Search for the AP value in the line using the pattern
ap_match = ap_pattern.search(line)

# Get the extracted AP value
if ap_match:
    ap_value = ap_match.group(1)
    print(f"Extracted AP Value: {ap_value}")
else:
    print("AP value not found in the line.")
