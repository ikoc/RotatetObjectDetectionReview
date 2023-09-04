import os
import datetime
import random 
from utils import *
from tqdm import tqdm
from PreparePlotGT import *
import cv2

current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
parent_directory = "/home/ekin/Desktop/workspace/RotatetObjectDetectionReview/figures/dotaTrainCheck/"
new_directory_path = os.path.join(parent_directory, timestamp)
try:
    os.makedirs(new_directory_path)
    print(f"Directory '{new_directory_path}' created successfully.")
except OSError as e:
    print(f"Error creating directory: {e}")


classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']   

rotated_AP = {}
for name in classnames:
    rotated_AP[name] = AngleGroup()

anno_path = r'/mnt/data/mmdata/DotaV1/train/labelTxt/{:s}.txt'
annotation_folder_path = os.path.dirname(anno_path)
file_list = os.listdir(annotation_folder_path)

random.shuffle(file_list)
file_list = file_list[0:1]
print(file_list)

for file_name in tqdm(file_list, desc="Processing files", unit="file"):
    file_path = os.path.join(annotation_folder_path, file_name)
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])

                object_struct['bbox'] = [float(splitlines[0]),
                                        float(splitlines[1]),
                                        float(splitlines[2]),
                                        float(splitlines[3]),
                                        float(splitlines[4]),
                                        float(splitlines[5]),
                                        float(splitlines[6]),
                                        float(splitlines[7])]  
                object_struct["rot_rectangle"] = convertToRotOpencv(object_struct['bbox'])

                rotated_AP[object_struct['name']].increment_gt(object_struct["rot_rectangle"]["calculated_angle"])
            else:
                break

rotated_AP = prepareData(rotated_AP)
runPlot(rotated_AP,["all"],new_directory_path,False)

for file_name in tqdm(file_list, desc="Processing files", unit="file"):
    print(file_name)
    image_path = "/mnt/data/mmdata/DotaV1/train/images/{}.png".format(file_name.split(".")[0])
    out_image_path = os.path.join(new_directory_path,file_name.split(".")[0]+".jpg")
    image = cv2.imread(image_path)
    file_path = os.path.join(annotation_folder_path, file_name)
    with open(file_path, 'r') as f:
        ground_truth_lines = f.readlines()

    ground_truth_polygons = []
    ground_truth_angles = []
    ground_truth_longest_edge = []
    for line in ground_truth_lines:
        line = line.strip()
        splitlines = line.split(' ')
        object_struct = {}
        if (len(splitlines) < 9):
            continue

        ground_truth_angles.append(calc_gt(line))
        ground_truth_longest_edge.append(get_longest_edge(line))
        parts = line.split()
        x = [int(parts[i]) for i in range(0, len(parts)-2, 2)]
        y = [int(parts[i+1]) for i in range(0, len(parts)-2, 2)]
        ground_truth_polygons.append(np.array(list(zip(x, y)), np.int32))

    i = 0
    for polygon in ground_truth_polygons:
        cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        pointA,pointB = ground_truth_longest_edge[i]
        cv2.line(image, tuple(pointA.astype(np.int32)),tuple(pointB.astype(np.int32)), color=(0, 0, 255), thickness=2)
        centroid = np.mean(polygon, axis=0).astype(int)
        centroid = (centroid[0],centroid[1]+40)
        text = "{}|{}".format(i,ground_truth_angles[i])
        print(text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
        bg_rect = (centroid[0],centroid[1]-text_size[1], text_size[0], text_size[1])
        text_color = (0, 0, 0) 
        bg_color = (255, 255, 255)  
        cv2.rectangle(image, bg_rect, bg_color, -1)
        cv2.putText(image, text, tuple(centroid), font, font_scale, text_color, 1, cv2.LINE_AA)
        i += 1
    
    #cv2.imshow('Image with Polygons', image)
    #cv2.waitKey(0)
    cv2.imwrite(out_image_path,image)
    cv2.destroyAllWindows()
