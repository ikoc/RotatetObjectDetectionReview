from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def drawCircularBarPlot(df,index,title=""):
    ax = plt.subplot(index, polar=True)
    ax.set_title(title)
    plt.axis('off')

    # Constants = parameters controling the plot layout:
    upperLimit = 100
    lowerLimit = 5
    labelPadding = 5

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
        edgecolor="white"
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
    
def drawBarPlot(df,index,title="",width=2):
    name = df['Name']
    count = df['Value']
    bx = plt.subplot(index)
    bx.bar(name, count,width=width)
    bx.set_xticks(name)
    bx.set_title(title)
    bx.set_xlabel("Angle")
    bx.set_ylabel("Sample Count")

def drawPieChart(df,index,title=""):
    name = df['Name']
    count = df['Value']   
    cx = plt.subplot(index)
    cx.pie(count, labels = name, startangle = 90)

data = loadRotatedAP("test_data/gt_angle_15degreeInterval.pickle")
data["all"] = AngleGroup()

clsList = data.keys()

for cls in clsList:
    if cls == "all":
        continue
    for index,value in enumerate(data[cls].gt):
        data["all"].gt[index] += value 
    for index,value in enumerate(data[cls].tp):
        data["all"].tp[index] += value
    for item in data["all"].angle.keys():
        data["all"].angle[item] += data[cls].angle[item] 

angleList = [angle for angle in range(-90,90,15)] 
first = angleList.pop(0)
angleList.append(first)

for cls in clsList:
    print(cls)
    content = {"Name":[],"Value":[]}
    for angle in angleList:
        content["Name"].append(angle) 
        content["Value"].append(data[cls].angle[angle])     
        
    df = pd.DataFrame(content)
    plt.figure(figsize=(10,10))
    drawCircularBarPlot(df,211,cls.upper() +" Angle Distrubition" )
    drawBarPlot(df,212)

    '''
    content = {"Name":["[-90,-45)","[-45,0)","[0,45)","[45,90)"],
               "Value":data[cls].gt}
    df = pd.DataFrame(content)
    drawPieChart(df,223) ## SINAN HOCA FeedBack GEREK YOK 
    drawBarPlot(df,224,width=0.4)
    '''

    plt.savefig("figures/gtDist/{}.png".format(cls))