import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('4_3_blackjack_training_data.csv')

# Confirm that 'Outcome' is in the DataFrame
if 'Outcome' in df.columns:
    # Calculate the win percentage
    column_data = df['Outcome']
    column_length = len(column_data)
    sum_of_wins = column_data.sum()
    win_percentage = (sum_of_wins / column_length) * 100
    print(f"Win percentage: {win_percentage:.2f}%")
else:
    print("Column 'Outcome' not found. Available columns:", df.columns)