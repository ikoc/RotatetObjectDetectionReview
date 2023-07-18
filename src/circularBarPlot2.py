from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(30,30))

data = loadRotatedAP("/home/ekin/Desktop/workspace/tez_ibrahim_koc/gt_angle.pickle")

content = {"Name":[],"Value":[]}
angleList = [angle for angle in range(-90,90,10)] 
first = angleList.pop(0)
angleList.append(first)

for cls in data.keys():
    for angle in angleList:
        content["Name"].append(angle) 
        content["Value"].append(data[cls].angle[angle])     
    break    
df = pd.DataFrame(content)


# initialize the figure
ax = plt.subplot(111, polar=True)
plt.axis('off')
plt.title(cls)

# Constants = parameters controling the plot layout:
upperLimit = 100
lowerLimit = 40
labelPadding = 10

# Compute max and min in the dataset
max = df['Value'].max()

# Let's compute heights: they are a conversion of each item value in those new coordinates
# In our example, 0 in the dataset will be converted to the lowerLimit (10)
# The maximum will be converted to the upperLimit (100)
slope = (max - lowerLimit) / max
heights = slope * df.Value + lowerLimit

# Compute the width of each bar. In total we have 2*Pi = 360°
width = 2*np.pi / len(df.index)

# Compute the angle each bar is centered on:
indexes = list(range(1, len(df.index)+1))
angles = [element * width for element in indexes]

# Draw bars
bars = ax.bar(
    x=angles, 
    height=heights, 
    width=width, 
    bottom=lowerLimit,
    linewidth=2, 
    edgecolor="white",
    color="#61a4b2",
)

# Add labels
for bar, angle, height, label in zip(bars,angles, heights, df["Name"]):
    # Labels are rotated. Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle)
    # Flip some labels upside down
    alignment = ""
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"

    # Finally add the labels
    ax.text(
        x=angle, 
        y=lowerLimit + bar.get_height() + labelPadding, 
        s=label, 
        ha=alignment, 
        va='center', 
        rotation=rotation, 
        rotation_mode="anchor") 
    

plt.savefig("test.png")