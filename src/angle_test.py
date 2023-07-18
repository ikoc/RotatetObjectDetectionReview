import math
import numpy as np

def polygonToRotRectangle(bbox):
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(2,4),order='F')
    angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    degree_value = angle * 180 / math.pi
    print("Arctangent of -({}-{}) / {}-{}  is {} -> mod90 = {}".format(bbox[0,1],bbox[0,0],bbox[1,1],bbox[1,0],degree_value, degree_value%90))

def printAllAngle(bbox):
    print("--------------------------")
    print(bbox)
    polygonToRotRectangle(bbox)
    polygonToRotRectangle(np.roll(bbox,2))
    polygonToRotRectangle(np.roll(bbox,4))
    polygonToRotRectangle(np.roll(bbox,6))

def testRoll(bbox):

    print(bbox)
    polygonToRotRectangle(bbox)
    polygonToRotRectangle(np.roll(bbox,2))
    polygonToRotRectangle(np.roll(bbox,4))
    polygonToRotRectangle(np.roll(bbox,6))

def shiftBboxPointsStartWithLowest(bbox):
    '''
    Author Ibrahim Koc
    Shift bbox values that lowest point 
    "which means highest y value point" will be starting poing x1 y1
    '''
    y_values = bbox[1::2]
    sorted_y_indexes= np.argsort(-y_values) ## find max y value
    max_y_index_of_bbox = (sorted_y_indexes[0] + 1) * 2 - 1 
    x_index_of_max_y = max_y_index_of_bbox -1
    bbox = np.roll(bbox,-x_index_of_max_y)
    return bbox

def testRoll2():

    gt_bbox = np.array([432., 985., 407., 972., 463., 882., 488., 902.])
    pred_bbox = np.array([465.01, 881.19, 490.07, 898.3 , 432.37, 982.8 , 407.31, 965.69])

    print("gt_bbox",gt_bbox)
    polygonToRotRectangle(gt_bbox)
    print("")
    print("gt_bbox_modified",shiftBboxPointsStartWithLowest(gt_bbox))
    polygonToRotRectangle(shiftBboxPointsStartWithLowest(gt_bbox))
    print("")

    print("pred_bbox",pred_bbox)
    polygonToRotRectangle(pred_bbox)
    print("")
    print("pred_bbox_modified",shiftBboxPointsStartWithLowest(pred_bbox))
    polygonToRotRectangle(shiftBboxPointsStartWithLowest(pred_bbox))
    print("")


    testRoll(pred_bbox)
                          
def test2Rotated():
    long_edge = np.array([465.01, 881.19, 490.07, 898.3 , 432.37, 982.8 , 407.31, 965.69])
    short_edge = np.array([0,10,100,100,110,90,10,0])

def testComparison():
    gt = np.array([432.0, 985.0, 407.0, 972.0, 463.0, 882.0, 488.0, 902.0])
    pred = np.array([465.01, 881.19, 490.07, 898.3, 432.37, 982.8, 407.31, 965.69])

    gt = shiftBboxPointsStartWithLowest(gt)
    pred = shiftBboxPointsStartWithLowest(pred)
    printAllAngle(gt)
    printAllAngle(pred)

import numpy as np

# Define the coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
coordinates = [20, 0, 40, 0, 40, 40, 20, 40]
points = np.array(coordinates)
points = np.append(points,points[0:2]) 
points = points.reshape((5, 2))
# Calculate the Euclidean distances between consecutive points
distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

# Find the index of the longest edge
longest_edge_index = np.argmax(distances)
longest_edge_length = distances[longest_edge_index]
longest_edge_vertices = points[longest_edge_index], points[longest_edge_index + 1]

print("Longest Edge Length:", longest_edge_length)
print("Longest Edge Vertices:", longest_edge_vertices)
