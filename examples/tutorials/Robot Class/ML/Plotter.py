import pandas as pd
import matplotlib.pyplot as plt

# Assuming data is in a CSV file called "data.csv"
data = pd.read_csv("OmegaII_6_6.csv")

plt.figure(figsize=(10,6))
plt.hist(data['init_hand'], bins=range(1,23), edgecolor='black', alpha=0.7)
plt.title('Histogram of Player\'s Initial Hand Values')
plt.xlabel('Initial Hand Value')
plt.ylabel('Frequency')
plt.xticks(range(1,22))
plt.show()

dealer_card_counts = data['dealer_card'].value_counts()

plt.figure(figsize=(10,6))
dealer_card_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Bar Chart of Dealer\'s Visible Cards')
plt.xlabel('Dealer Card')
plt.ylabel('Frequency')
plt.show()

hit_counts = data['hit'].value_counts()

plt.figure(figsize=(10,6))
hit_counts.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Bar Chart Showing Number of Hits vs Stays by Players')
plt.xlabel('Action (Hit/Stay)')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()
