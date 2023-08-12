from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def drawCircularBarPlot(df, index, title=""):
    ax = plt.subplot(index, polar=True)
    ax.set_title(title)
    plt.axis('off')

    # Constants = parameters controling the plot layout:
    upperLimit = 100
    lowerLimit = 100
    labelPadding = 50

    # Compute max and min in the dataset
    max = df['Value'].max()

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max - lowerLimit) / max
    heights = slope * df.Value + lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    #width = np.pi*2 / len(df.index)
    width = np.pi / len(df.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df.index) + 1)) 
    #angles = [element * width for element in indexes]
    angles = [element * width + np.pi*3/2 - np.pi/12 for element in indexes]
    print(angles)

    maxCount = df['Value'].max()
    minCount = df['Value'].min()
    GROUPS_SIZE = 5
    value_range = (maxCount - minCount) / GROUPS_SIZE
    df['ColorGroup'] = pd.cut(df['Value'], bins=GROUPS_SIZE, labels=False)
    COLORS = [f"C{i}" for i in df['ColorGroup']]

    print(df)
    print(COLORS)

    # Draw bars
    bars = ax.bar(
        x=angles,
        height=heights,
        width=width,
        bottom=lowerLimit,
        linewidth=2,
        edgecolor="white",
        color=COLORS
    )

    # Add labels
    for bar, angle, height, angle_group,angle_count in zip(bars, angles, heights, df["Name"], df["Value"]):
        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)
        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle,
            y=(bar.get_height() + labelPadding)/2,
            s="{}".format(angle_count),
            ha=alignment,
            va='center',
            rotation=rotation,
            rotation_mode="anchor"
        )

        ax.text(
            x=angle,
            y=lowerLimit + (bar.get_height() + labelPadding),
            s="{}{}".format(angle_group,chr(176)),
            ha=alignment,
            va='center',
            rotation=rotation,
            rotation_mode="anchor"
        )
    COLORS_LEGEND = [f"C{i}" for i in range(GROUPS_SIZE)]
    legend_labels = [f"{minCount + i * value_range:.1f}-{minCount + (i + 1) * value_range:.1f}" for i in range(GROUPS_SIZE)]
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(COLORS_LEGEND, legend_labels)]
    #ax.legend(handles=legend_patches, title='Sample Count Intervals', loc=(-0.1,0.75))
    ax.legend(handles=legend_patches, title='Sample Count Intervals', loc=(0,0.35))

    #legend_labels = [f"{minCount + i * value_range:.1f}-{minCount + (i + 1) * value_range:.1f}" for i in range(GROUPS_SIZE)]
    #ax.legend(bars, legend_labels, title='Value Range', loc='upper left')



def drawBarPlot(df, index, title="", width=2):
    name = df['Name']
    count = df['Value']
    bx = plt.subplot(index)
    bx.bar(name, count, width=width)
    bx.set_xticks(name)
    bx.set_title(title)
    bx.set_xlabel("Angle")
    bx.set_ylabel("Sample Count")


def drawPieChart(df, index, title=""):
    name = df['Name']
    count = df['Value']
    cx = plt.subplot(index)
    cx.pie(count, labels=name, startangle=90)


data = loadRotatedAP("/home/ekin/Desktop/workspace/RotatetObjectDetectionReview/DotaTrainAnalysisResult/data.pickle")
data["all"] = AngleGroup()

clsList = data.keys()

for cls in clsList:
    if cls == "all":
        continue
    for index, value in enumerate(data[cls].gt):
        data["all"].gt[index] += value
    for index, value in enumerate(data[cls].tp):
        data["all"].tp[index] += value
    for item in data["all"].angle.keys():
        data["all"].angle[item] += data[cls].angle[item]

angleList = [angle for angle in range(-90, 90, 15)]
#first = angleList.pop(0)
#angleList.append(first)

for cls in clsList:
    print(cls)
    content = {"Name": [], "Value": []}
    for angle in angleList:
        content["Name"].append(angle)
        content["Value"].append(data[cls].angle[angle])

    df = pd.DataFrame(content)
    plt.figure(figsize=(6, 6))
    drawCircularBarPlot(df, 111, cls.upper() + " Angle Distrubition")
    #drawBarPlot(df, 212)

    '''
    content = {"Name":["[-90,-45)","[-45,0)","[0,45)","[45,90)"],
               "Value":data[cls].gt}
    df = pd.DataFrame(content)
    drawPieChart(df,223) ## SINAN HOCA FeedBack GEREK YOK 
    drawBarPlot(df,224,width=0.4)
    '''

    plt.savefig("/home/ekin/Desktop/workspace/RotatetObjectDetectionReview/figures/gtDist/dota_train_{}.png".format(cls))
