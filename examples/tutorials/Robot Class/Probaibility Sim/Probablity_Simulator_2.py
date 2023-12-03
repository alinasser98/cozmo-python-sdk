import random
import pandas as pd
import math

# Create parameters
NUM_DECKS = 8 # Number of decks in the shoe
NUM_PLAYERS = 4 # Number of players at the table
NUM_ROUNDS = 10000 # Number of rounds to simulate
MIN_CARDS_BEFORE_RESHUFFLE = 52 # Minimum number of cards before reshuffling

# Initialize the creation of the deck
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades'] # Suits in a deck
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] # Ranks in a deck
DECK = [(rank, suit) for suit in SUITS for rank in RANKS] # This created a list of tuples for each card in the deck

# Function to create and shuffle a new shoe
def create_shuffled_shoe(num_decks): # I am defining a function to create a shuffled shoe
    shoe = DECK * num_decks # this creates a shoe with the number of decks specified
    random.shuffle(shoe) # this shuffles the shoe using the random library
    return shoe

# Creating a shoe
shoe = create_shuffled_shoe(NUM_DECKS) # this creates a shoe with the number of decks specified

# Initialize card count
card_count = {card: 0 for card in RANKS} # card_count = {'2':0,'3':0,.....,'A':0}

# Function to calculate the hand value of a player
def calculate_hand_value(hand): # I am defining a function to calculate the hand value
    value, aces = 0, 0 # this sets the value and aces to 0
    for card in hand: # this loops through the hand
        rank, _ = card # this sets the rank to the first value in the card tuple
        if rank in ['J', 'Q', 'K']: # this checks if the rank is a face card
            value += 10 # this adds 10 to the value variable
        elif rank == 'A': # this checks if the rank is an ace
            aces += 1 # this adds 1 to the aces variable
        else:
            value += int(rank) # this adds the rank to the value variable
    for _ in range(aces): # this loops through the number of aces
        value += 11 if value + 11 <= 21 else 1 # this adds 11 to the value if the value is less than 21, otherwise it adds 1
    return value

# Update Card Count Function
def update_card_count(card): # I am defining a function to update the card count
    rank, _ = card
    card_count[rank] += 1 # this adds 1 to the card count for the rank

# Probability Calculation Function
def calculate_probabilities(num_decks_remaining, card_count): # I am defining a function to calculate the probabilities
    total_cards = num_decks_remaining * 52 # using this to calculate the total number of cards remaining
    probabilities = {} #I am creating a dictionary to store and track the probabilities
    for card in RANKS: # this loops through the ranks
        count_of_card = card_count.get(card, 0) # this gets the count of the card from the card_count dictionary and sets it to 0 if it doesn't exist
        probabilities[card] = count_of_card / total_cards if total_cards > 0 else 0 # this calculates the probability of the card and stores it in the dictionary
    return probabilities # this returns the probabilities dictionary

# Decision Making Function
def make_decision(hand, dealer_card, num_decks_remaining): # I am defining a function to make a decision for weather a player should hit or stay
    hand_value = calculate_hand_value(hand) # this calculates the hand value of the player
    probabilities = calculate_probabilities(num_decks_remaining, card_count) # this calculates the probabilities of the cards remaining
    # Calculate probability of a safe hit (not exceeding 21)
    safe_hit_prob = 0 # I created a variable to store the probability of a safe hit
    for rank, prob in probabilities.items(): # this loops through the probabilities
        if rank in ['J', 'Q', 'K']: # this checks if the rank is a face card
            added_value = 10 # this sets the added value to 10 if it is a face card
        elif rank == 'A': # this checks if the rank is an ace
            added_value = 11 if hand_value + 11 <= 21 else 1 # this sets the added value to 11 if the hand value is less than 21, otherwise it sets it to 1
        else:
            added_value = int(rank) # this translates the rank that was a string to an integer and sets it to the added value
        if hand_value + added_value <= 21: # this checks if the hand value plus the added value is less than or equal to 21
            safe_hit_prob += prob # I am using this to add the probability of a safe hit to the safe_hit_prob variable
    if hand_value < 12 or (hand_value < 21 and safe_hit_prob > 0.5): # this checks if the hand value is less than 12 or if the hand value is less than 21 and the safe hit probability is greater than 0.5
        return 1  # If it returns 1 the player will hit
    else:
        return 0  # If it returns 0 the player will stay

# Data Collection
data = [] # this is for data storage

# Game Simulation Loop (for each round)
for _ in range(NUM_ROUNDS): # this loops through the number of rounds specified in the parameters that were set earlier
    # Check how many decks are remaining and round up
    num_decks_remaining = math.ceil(len(shoe) / (52 * NUM_DECKS)) # this calculates the number of decks remaining and rounds up
                                                                  # math.ceil() rounds up
    # Check and reshuffle the shoe if necessary
    if len(shoe) < MIN_CARDS_BEFORE_RESHUFFLE: # this checks if the number of cards in the shoe is less than the minimum cards before reshuffle
        shoe = create_shuffled_shoe(NUM_DECKS) # this creates a new shoe with the number of decks specified
        card_count = {card: 0 for card in RANKS} # this resets the card count to 0 for each card
    player_hands = [[] for _ in range(NUM_PLAYERS)] # this creates a list of lists for each player hand
    dealer_hand = [shoe.pop()]  # Dealer's face-down card

    # Deal first card to each player
    for player_hand in player_hands: # this loops through the player hand list
        card = shoe.pop() # this pops the first card from the shoe and sets it to the card variable
        player_hand.append(card) # this appends the card to the player hand list
        update_card_count(card) # this updates the card count for the card that was popped from the shoe

    # Dealer's face-up card
    dealer_card = shoe.pop() # this pops the first card from the shoe and sets it to the dealer_card variable
    dealer_hand.append(dealer_card) # this appends the dealer card to the dealer hand list
    update_card_count(dealer_card) # this updates the card count for the card that was popped from the shoe

    # Deal second card to each player
    for player_hand in player_hands:
        card = shoe.pop() 
        player_hand.append(card) # this appends the second card to the player hand list
        update_card_count(card)

    # The players make their decisions
    for player_hand in player_hands: 
        action = make_decision(player_hand, dealer_card, num_decks_remaining) # this calls the make_decision function and passes the player hand, dealer card, and number of decks remaining
        if action == 1:  # Hit
            player_hand.append(shoe.pop()) # this appends the popped card to the player hand list
            update_card_count(player_hand[-1]) # this updates the card count for the card that was popped from the shoe
        player_value = calculate_hand_value(player_hand) # this calculates the hand value of the player
        dealer_value = calculate_hand_value(dealer_hand) # this calculates the hand value of the dealer
        if dealer_value > 21 or player_value > dealer_value: # this checks if the dealer value is greater than 21 or if the player value is greater than the dealer value
            outcome = 1  # A win
        elif player_value == dealer_value: # this checks if the player value is equal to the dealer value
            outcome = 0  # A tie
        else:
            outcome = -1  # A loss
        update_card_count(dealer_hand[0]) # this updates the card count for the dealer hand face down card since we are done with the round
        current_probabilities = calculate_probabilities(num_decks_remaining, card_count) # this calculates the probabilities of the cards remaining in the shoe after the round

        # this is how I am storing the data for each round
        round_data = {
            'num_players': NUM_PLAYERS, # this is the number of players at the table
            'num_decks': NUM_DECKS, # this is the number of decks in the shoe
            'Player Initial Hand': calculate_hand_value(player_hand[:2]), # this is the player initial hand value (first 2 cards)
            'Dealer Up Card': calculate_hand_value([dealer_card]), # this is the dealer up card value
            'Action Taken': action, # this is the action taken by the player (hit or stay)
            'Outcome': outcome # this is the outcome of the round (win, loss, or tie)
        }
        for card in RANKS:
            round_data[f'Count_{card}'] = card_count[card] # this appends the card count for each card to the round data
            round_data[f'Probability_{card}'] = current_probabilities[card] # this appends the probability for each card to the round data

        # Appending round data to the data list
        data.append(round_data) # this appends the round data to the data list

# Export to CSV after all rounds are complete
df = pd.DataFrame(data) # this creates a dataframe from the data list
df.to_csv(f"{NUM_DECKS}_{NUM_PLAYERS}_blackjack_training_data.csv", index=False) # this exports the dataframe to a csv file
