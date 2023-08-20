import matplotlib.pyplot as plt
import numpy as np


# Precision and Recall data
precision = [0, 1/2, 2/3, 3/4, 4/5, 5/6, 5/7, 5/8, 5/9, 5/10]
recall = [0, 1/5, 2/5, 3/5, 4/5, 5/5, 5/5, 5/5, 5/5, 5/5]

#precision = precision[0:6]
#recall = recall[0:6]

# Calculate the area under the precision-recall curve using the trapezoidal rule
area_under_curve = np.trapz(precision, recall)
print("Area under the Precision-Recall curve:", area_under_curve)

# Plot the precision-recall curve
plt.plot(recall, precision, marker='o')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()
plt.ylim(0, 1.1)
plt.xlim(0, 1.1)

# Annotate points with their corresponding BBox IDs
for i, bbox_id in enumerate(range(len(recall))):
    plt.annotate(bbox_id, (recall[i], precision[i]), textcoords="offset points", xytext=(5,5), ha='center')

plt.show()