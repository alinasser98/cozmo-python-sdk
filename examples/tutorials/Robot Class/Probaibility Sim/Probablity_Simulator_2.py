import random
import pandas as pd
import math

# Create parameters
NUM_DECKS = 4 # Number of decks in the shoe
NUM_PLAYERS = 3 # Number of players at the table
NUM_ROUNDS = 1000 # Number of rounds to simulate
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
for _ in range(NUM_ROUNDS):
    # Check how many decks are remaining and round up
    num_decks_remaining = math.ceil(len(shoe) / (52 * NUM_DECKS))

    # Check and reshuffle the shoe if necessary
    if len(shoe) < MIN_CARDS_BEFORE_RESHUFFLE:
        shoe = create_shuffled_shoe(NUM_DECKS)
        card_count = {card: 0 for card in RANKS}
    player_hands = [[] for _ in range(NUM_PLAYERS)]
    dealer_hand = [shoe.pop()]  # Dealer's face-down card

    # Deal first card to each player
    for player_hand in player_hands:
        card = shoe.pop()
        player_hand.append(card)
        update_card_count(card)

    # Dealer's face-up card
    dealer_card = shoe.pop()
    dealer_hand.append(dealer_card)
    update_card_count(dealer_card)

    # Deal second card to each player
    for player_hand in player_hands:
        card = shoe.pop()
        player_hand.append(card)
        update_card_count(card)

    # Player's decisions and outcomes
    for player_hand in player_hands:
        action = make_decision(player_hand, dealer_card, num_decks_remaining)
        if action == 1:  # Hit
            player_hand.append(shoe.pop())
            update_card_count(player_hand[-1])
        player_value = calculate_hand_value(player_hand)
        dealer_value = calculate_hand_value(dealer_hand)
        if dealer_value > 21 or player_value > dealer_value:
            outcome = 1  # Win
        elif player_value == dealer_value:
            outcome = 0  # Tie
        else:
            outcome = -1  # Lose 
        update_card_count(dealer_hand[0])  # Now count dealer's face-down card
        # Record data
        data.append({
            'num_players': NUM_PLAYERS,
            'num_decks': NUM_DECKS,
            'Player Initial Hand': calculate_hand_value(player_hand[:2]),
            'Dealer Up Card': calculate_hand_value([dealer_card]),
            'Action Taken': action,
            'Outcome': outcome
        })
# Export to CSV after all rounds are complete
df = pd.DataFrame(data)
df.to_csv(f"{NUM_DECKS}_{NUM_PLAYERS}_blackjack_training_data.csv", index=False)
