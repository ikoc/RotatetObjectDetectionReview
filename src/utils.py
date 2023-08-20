import cv2
import pickle
import numpy as np

class AngleGroup():
    '''
        GT means Ground truth sample number in dataset.
        TP means True Positive prediction number in dataset.
    '''
    def __init__(self) -> None:        
        self.gt = [0,0,0,0]   ## [-90,-45) [-45,0) [0,45) [45,90)
        self.tp = [0,0,0,0]   
        self.angle = {}
        for i in range(-90,90,15):
            self.angle[i] = 0

    def get_ap(self):
        return (self.get_tp()/self.get_gt())

    def get_gt(self):
        return np.array(self.gt)

    def get_tp(self):
        return np.array(self.tp)

    def increment(self,angle,type):
        angle_group = round(angle/15)*15
        if angle_group == 90:
            angle_group = -90
        self.angle[angle_group] += 1

        if angle >= -90 and angle < -45:
            index = 0
        elif angle >= -45 and angle < 0:
            index = 1
        elif angle >= 0 and angle < 45:
            index = 2
        elif angle >=45 and angle < 90:
            index = 3        
        elif angle == 90:
            index = 0
        else:
            print("Invalid angle:{}".format(angle))
            return
        
        if type == "gt":
            self.gt[index] += 1
        else:
            self.tp[index] += 1

    def increment_gt(self,angle):
        self.increment(angle,"gt")
    def increment_tp(self,angle):
        self.increment(angle,"tp")
    def __repr__(self) -> str:
        return "GT:{} TP:{}".format(self.gt,self.tp)
    
def dumpRotatedAP(rotatedAP):
    with open('rotatedAP.pickle', 'wb') as handle:
        pickle.dump(rotatedAP, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadRotatedAP(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b        

def calculateAngle(start, end):
    #print("Point end: {} - Point start: {}".format(end,start))
    ## Multiple by -1 , because of Y axis getting bigger when goes down. 
    angle_rad = np.arctan2( -1*(end[1] - start[1]) , end[0] - start[0]) 
    angle_deg = np.degrees(angle_rad)
    #print("Angle (in radians):", angle_rad)
    #print("Calculated Angle (in degrees):", angle_deg)
    return angle_deg

def calculateAngleLongestEdge(coordinates,getWithGroup=False):
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
    if getWithGroup:
        return "{}|{:.2f}".format(get_angle_group(theta),theta)
    return theta

def getLongestEdge(coordinates):
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
    return (pointA , pointB)

def get_angle_group(angle):
    angle_group = round(angle/15)*15
    if angle_group == 90:
        angle_group = -90
    return angle_group

def calc_gt(line):
    points = line.strip().split()[:8]
    points = [float(p) for p in points]
    return calculateAngleLongestEdge(points,True)

def calc_det(line):
    points = line.strip().split()[2:]
    points = [float(p) for p in points]
    return calculateAngleLongestEdge(points,True)

def get_longest_edge(line):
    points = line.strip().split()[:8]
    points = [float(p) for p in points]
    return getLongestEdge(points)


def convertToRotOpencv(coordinate,longEdgeVersionActive=True):
    """
    Author Ibrahim Koc
    If longEdgeVersionActive it will check width > height 
                            than angle will be multiplied by -1
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: format [x_c, y_c, w, h, theta]
    """
    areaOfPolygon = -1 # calculate_polygon_area(coordinate)
    box = np.int0(coordinate)
    box = box.reshape([4, 2])
    rect1 = cv2.minAreaRect(box)

    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
    if longEdgeVersionActive:
        theta = calculateAngleLongestEdge(coordinates=coordinate)
        
    if theta == 90:
        theta = -90
    
    myDict = {"center_x":float(x),
              "center_y":float(y),
              "w":w,
              "h":h,
              "calculated_angle":theta,
              "bbox":coordinate,
              "longEdge":longEdgeVersionActive ,
              "area":areaOfPolygon}
    return myDict