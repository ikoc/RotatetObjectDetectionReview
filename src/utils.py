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
        for i in range(-90,90,10):
            self.angle[i] = 0

    def get_ap(self):
        return (self.get_tp()/self.get_gt())

    def get_gt(self):
        return np.array(self.gt)

    def get_tp(self):
        return np.array(self.tp)

    def increment(self,angle,type):
        angle_group = round(angle/10)*10
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
