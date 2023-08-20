import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from PreparePlotGT import drawCircularBarPlot

class ApResult():
    def __init__(self,name) -> None:
        self.name = name
        self.ap = {"area":{},"angle":{}}

    def drawAnglePlot(self):
        angleList = [angle for angle in range(-90, 90, 15)]
        content = {"Name": [], "Value": []}
        for angle in angleList:
            content["Name"].append(angle)
            content["Value"].append(self.ap["angle"][angle])
        df = pd.DataFrame(content)
        drawCircularBarPlot(df, 111, self.name.upper() + " Angle Distrubition")
        plt.show()


# Read the content from the file
file_path = "/home/ekin/Desktop/workspace/RotatetObjectDetectionReview/test_data/dotaValResults.txt"  # Replace with the actual file path
with open(file_path, "r") as file:
    lines = file.readlines()

result_list = []

# Define regular expressions to match the values
category_pattern = re.compile(r"Category name\s+(\w+)")
angle_pattern = re.compile(r"angle=\s*(-?\d+)\s*")
area_pattern = re.compile(r"area=\s*(\w+)\s*")
ap_pattern = re.compile(r"= (\d+\.\d+)")


for line in lines:
    line = line.strip()
    if "Category name" in line:
        category_match = category_pattern.search(line)
        category_name = category_match.group(1) if category_match else None
        print(category_name)
        result_list.append(ApResult(category_name))
    if "Average Precision" in line and "maxDets=500" in line:
        area_match = area_pattern.search(line)
        angle_match = angle_pattern.search(line)
        ap_match = ap_pattern.search(line)
        area = area_match.group(1) if area_match else None
        angle = angle_match.group(1) if angle_match else "all"
        ap = ap_match.group(1) if ap_match else -1

        if angle != "all":
            angle = int(angle)
        # Print the extracted values
        #print(f"Area: {area}")
        #print(f"Angle: {angle}")
        #print(f"AP: {ap}")
        if area not in result_list[-1].ap["area"]: 
            result_list[-1].ap["area"][area] = {}
        result_list[-1].ap["area"][area][angle] = ap

        if angle not in result_list[-1].ap["angle"]: 
            result_list[-1].ap["angle"][angle] = {}
        result_list[-1].ap["angle"][angle][area] = ap

for r in result_list:
    r.drawAnglePlot()
