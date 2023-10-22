import pandas as pd
import matplotlib.pyplot as plt

# Assuming data is in a CSV file called "data.csv"
df = pd.read_csv('Data.csv') # Use pd.read_csv for CSV files

# Filter for rows where outcome is 1 (win)
win_df = df[df['outcome'] == 1]

# Calculate the probability for each initial hand value
prob_init_hand = win_df.groupby('init_hand').size() / df.groupby('init_hand').size()

# Plotting
prob_init_hand.plot(kind='bar', color='skyblue')
plt.title('Probability of Winning Based on Initial Hand Value')
plt.ylabel('Probability')
plt.xlabel('Initial Hand Value')
plt.show()

# Calculate the probability for each dealer's visible card
prob_dealer_card = win_df.groupby('dealer_card').size() / df.groupby('dealer_card').size()

# Plotting
prob_dealer_card.plot(kind='bar', color='salmon')
plt.title('Probability of Winning Based on Dealer\'s Visible Card')
plt.ylabel('Probability')
plt.xlabel('Dealer\'s Visible Card')
plt.show()
