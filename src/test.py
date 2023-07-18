import matplotlib.pyplot as plt

def pie():
    # Data for the pie chart
    labels = ['Category 1', 'Category 2', 'Category 3']
    values = [30, 50, 20]

    # Colors for the pie chart
    colors = ['red', 'green', 'blue']

    # Explode the pie chart (optional)
    explode = (0.1, 0, 0)

    # Create the pie chart
    plt.pie(values, labels=labels, colors=colors, explode=explode, startangle=90, shadow=True)

    # Add a title
    plt.title('Pie-Shaped Bar Chart')

    # Display the chart
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
values = [25, 30, 15, 20]

# Calculate total value
total = sum(values)

# Calculate percentage for each category
percentages = [value / total * 100 for value in values]

# Create bar chart
fig, ax = plt.subplots()
bars = ax.bar(range(len(categories)), percentages, width=0.6)

# Set colors for bars
colors = ['red', 'green', 'blue', 'yellow']
for i, bar in enumerate(bars):
    bar.set_color(colors[i])

# Add labels
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories)

# Add percentage labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{percentages[i]:.1f}%', ha='center', va='bottom')

# Remove y-axis labels and ticks
ax.set_ylabel('Percentage')
ax.yaxis.set_ticklabels([])

plt.show()
