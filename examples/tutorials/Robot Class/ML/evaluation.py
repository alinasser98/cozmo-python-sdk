import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout


# Read the CSV file into a DataFrame
df1 = pd.read_csv('OmegaII_6_6.csv')

column_data = df1['result']
column_length = len(column_data)
# Check if each value in the column is equal to 1
is_equal_to_1 = column_data == 1

# Filter the DataFrame to select rows where the values are equal to 1
filtered_data = df1[is_equal_to_1]

# Calculate the sum of the values that are equal to 1
sum_of_ones = filtered_data['result'].sum()


ali_win = (sum_of_ones/column_length)*100

df2 = pd.read_csv('Denise_blackjackdata_6_6.csv')

column_data = df2['result']
column_length = len(column_data)
# Check if each value in the column is equal to 1
is_equal_to_1 = column_data == 1

# Filter the DataFrame to select rows where the values are equal to 1
filtered_data = df2[is_equal_to_1]

# Calculate the sum of the values that are equal to 1
sum_of_ones = filtered_data['result'].sum()

denise_win = (sum_of_ones/column_length)*100

print(f"Ali win 6 6 % {ali_win}")
print(f"Denise win 6 6 % {denise_win}")


##########################################################
#   calculating win percentage for 4 players and 4 decks #
##########################################################


# df1 = pd.read_csv('ali_4_4.csv')

# column_data = df1['result']
# column_length = len(column_data)
# # Check if each value in the column is equal to 1
# is_equal_to_1 = column_data == 1

# # Filter the DataFrame to select rows where the values are equal to 1
# filtered_data = df1[is_equal_to_1]

# # Calculate the sum of the values that are equal to 1
# sum_of_ones = filtered_data['result'].sum()


# ali_win = (sum_of_ones/column_length)*100

# df2 = pd.read_csv('denise_4_4.csv')

# column_data = df2['result']
# column_length = len(column_data)
# # Check if each value in the column is equal to 1
# is_equal_to_1 = column_data == 1

# # Filter the DataFrame to select rows where the values are equal to 1
# filtered_data = df2[is_equal_to_1]

# # Calculate the sum of the values that are equal to 1
# sum_of_ones = filtered_data['result'].sum()

# denise_win = (sum_of_ones/column_length)*100

# print(f"Ali win 4 4 % {ali_win}")
# print(f"Denise win 4 4 % {denise_win}")