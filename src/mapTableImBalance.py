import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *

# +90 is same as -90 , all converted to -90.

columns = ('[-90,-45)', '[-45,0)', '[0,45)', '[45,90)')

rows = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

rows = ['baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
                'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool']

data = np.random.random((len(rows),len(columns)))

rotatedApData = loadRotatedAP("/home/ekin/Desktop/workspace/tez_ibrahim_koc/rotatedAP.pickle")

# Create an empty DataFrame with the columns and rows
data = pd.DataFrame(index=rows, columns=columns)
# Fill the DataFrame with average precision values (you need to replace the NaN values with the actual values)
for cls in rows:
    data.loc[cls] = rotatedApData[cls].get_gt()

# Create a figure and axis
fig, ax = plt.subplots()

# Hide axis
ax.axis('off')

# Create the table
table = ax.table(cellText=data.values, colLabels=data.columns, rowLabels=data.index, loc='center', cellLoc='center')

# Set table properties
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

# Set table cell colors
for i in range(len(rows)):
    for j in range(len(columns)):
        cell = table.get_celld()[i+1, j]
        if i % 2 == 0:
            cell.set_facecolor('#e6e6e6')
        else:
            cell.set_facecolor('#f2f2f2')


# Set table column widths
table.auto_set_column_width([0, 1, 2, 3])

# Set table title
title = "Distribution of Angle"
table_title = ax.set_title(title)

# Adjust layout
plt.tight_layout()

# Show the table
# plt.show()

plt.savefig(title+'.png',bbox_inches='tight')