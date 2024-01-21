import matplotlib.pyplot as plt

# Data from the image
categories = ['speed_40', 'speed_80', 'speed_50', 'speed_30', 'speed_100', 'speed_70', 'speed_120', 'end_speed']
values = [2015, 264, 70, 777, 286, 544, 478, 68]

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
