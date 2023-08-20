import cv2
import numpy as np

def calculateAngle(start, end):
    #print("Point end: {} - Point start: {}".format(end,start))
    ## Multiple by -1 , because of Y axis getting bigger when goes down. 
    angle_rad = np.arctan2( -1*(end[1] - start[1]) , end[0] - start[0]) 
    angle_deg = np.degrees(angle_rad)
    #print("Angle (in radians):", angle_rad)
    #print("Calculated Angle (in degrees):", angle_deg)
    return angle_deg

def calculateAngleLongestEdge(coordinates):
    # Append first point again to calculate lengths of all edges
    points = np.array(coordinates)
    points = np.append(points,points[0:2]) 
    points = points.reshape((5, 2))
    # Calculate the Euclidean distances between consecutive points
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # Find the index of the longest edge
    longest_edge_index = np.argmax(distances)
    longest_edge_length = distances[longest_edge_index]
    pointA , pointB = points[longest_edge_index], points[longest_edge_index + 1]
    # Select lower x values as Start point 
    if pointA[0] < pointB[0]:
        theta = calculateAngle(pointA, pointB)
    else:
        theta = calculateAngle(pointB, pointA)
    
    if theta == 90:
        theta = -90
    #print("Calculated Angle (in degrees):", theta)
    return "{},{:.2f}".format(get_angle_group(theta),theta)

def get_angle_group(angle):
    angle_group = round(angle/15)*15
    if angle_group == 90:
        angle_group = -90
    return angle_group

def calc_gt(line):
    points = line.strip().split()[:8]
    points = [float(p) for p in points]
    return calculateAngleLongestEdge(points)

def calc_det(line):
    points = line.strip().split()[2:]
    points = [float(p) for p in points]
    return calculateAngleLongestEdge(points)

# Read the image
image_path = "/home/ekin/Desktop/workspace/RotatetObjectDetectionReview/mapTestByHand/gt/images/P0348.png"
image = cv2.imread(image_path)

# Ground truth polygons from the label file
ground_truth_lines = [
    '31 1455 49 1365 446 1438 429 1532 harbor 0',
    '59 1118 70 1062 487 1107 479 1165 harbor 0',
    '213 715 229 677 506 767 488 804 harbor 0',
    '273 550 285 518 544 605 533 634 harbor 0',
    '369 225 390 171 547 227 526 278 harbor 0'
]

# Detection polygons from the detection file
detection_lines = [
    'P0348 0.9973205924034119 571.08 545.71 537.73 641.59 267.95 547.75 301.30 451.87',
    'P0348 0.9953234791755676 477.90 755.78 464.48 797.24 211.34 715.29 224.76 673.83',
    'P0348 0.995096743106842 485.37 1095.10 479.46 1164.35 59.45 1128.47 65.37 1059.22',
    'P0348 0.9818832874298096 563.21 610.74 552.54 642.55 273.56 548.90 284.24 517.10',
    'P0348 0.9585938453674316 462.25 1450.80 433.11 1583.53 -14.89 1485.17 14.25 1352.44',
    'P0348 0.915816605091095 565.69 229.96 551.69 277.87 331.88 213.64 345.88 165.73',
    'P0348 0.4810445010662079 661.72 1014.43 660.40 1029.87 544.89 1020.00 546.21 1004.55',
    'P0348 0.36773884296417236 110.93 1503.36 107.65 1521.55 -3.43 1501.52 -0.15 1483.34',
    'P0348 0.19812804460525513 712.02 358.27 701.20 375.61 555.53 284.70 566.35 267.36',
    'P0348 0.08572091907262802 759.03 387.32 746.15 406.91 650.01 343.73 662.89 324.14'
]

# Extract ground truth polygons
ground_truth_polygons = []
ground_truth_angles = []
for line in ground_truth_lines:
    ground_truth_angles.append(calc_gt(line))
    parts = line.split()
    x = [int(parts[i]) for i in range(0, len(parts)-2, 2)]
    y = [int(parts[i+1]) for i in range(0, len(parts)-2, 2)]
    ground_truth_polygons.append(np.array(list(zip(x, y)), np.int32))

print("GT angle {}".format(ground_truth_angles))
# Extract detection polygons
detection_polygons = []
detection_angles = []
for line in detection_lines:
    detection_angles.append(calc_det(line))
    parts = line.split()
    x = [float(parts[i]) for i in range(2, len(parts), 2)]
    y = [float(parts[i]) for i in range(3, len(parts), 2)]
    detection_polygons.append(np.array(list(zip(x, y)), np.int32))
print("Det angle {}".format(detection_angles))


# Draw ground truth polygons in green
i = 0
for polygon in ground_truth_polygons:
    cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=3)
    centroid = np.mean(polygon, axis=0).astype(int)
    centroid = (centroid[0],centroid[1]+40)
    cv2.putText(image, "{}".format(ground_truth_angles[i]), tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    i += 1

# Draw detection polygons in red
i = 0
for polygon in detection_polygons:
    cv2.polylines(image, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
    centroid = np.mean(polygon, axis=0).astype(int)
    cv2.putText(image, "{}:{}".format(i,detection_angles[i]), tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    i += 1

# Display the image with polygons
cv2.imshow('Image with Polygons', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# If you want to save the modified image
output_image_path = '/home/ekin/Desktop/workspace/RotatetObjectDetectionReview/mapTestByHand/modified_image_angle.jpg'
cv2.imwrite(output_image_path, image)
