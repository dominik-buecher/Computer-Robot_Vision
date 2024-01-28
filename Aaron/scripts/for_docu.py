import matplotlib.pyplot as plt

# Data from the image
categories = ['speed_120', 'speed_50', 'no_sign', 'speed_40', 'speed_100', 'speed_80', 'speed_30', 'speed_70', 'end_speed']
values = [1468, 1418, 8928, 2034, 1303, 1409, 1721, 1630, 694]

# Creating the bar plot
plt.figure(figsize=(10, 5))
bars = plt.bar(categories, values, color='blue')

# Adding the text labels above the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, yval, ha='center', va='bottom', fontsize=12)

# Setting the title and labels
plt.title('Anzahl Bilder pro Klasse',fontsize=16)
plt.xlabel('Klasse', fontsize=14)
plt.ylabel('# Bilder', fontsize=14)

# Increasing the fontsize of the tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show plot
plt.show()
