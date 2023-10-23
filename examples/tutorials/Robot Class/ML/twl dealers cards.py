import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv("Denise_blackjackdata_6_6.csv")

# Group by the dealer's card and then calculate the proportion of outcomes
grouped_data = data.groupby('dealer_card')['outcome'].value_counts(normalize=True).unstack().fillna(0)

# Ensure all columns are present in the grouped data
for outcome in [-1, 0, 1]:
    if outcome not in grouped_data.columns:
        grouped_data[outcome] = 0

# Create a stacked bar chart
plt.figure(figsize=(10,6))

grouped_data.sort_index(ascending=False).plot(kind='bar', stacked=True, color=['red', 'gray', 'green'], edgecolor='black', figsize=(10,6))
plt.title('Probability of Ties, Wins, and Losses based on Dealer\'s Card')
plt.ylabel('Probability')
plt.xlabel('Dealer\'s Card')
plt.legend(["Loss", "Tie", "Win"], loc='upper left')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
