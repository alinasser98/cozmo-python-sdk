import random
import pandas as pd

# Parameters
NUM_DECKS = 4
NUM_PLAYERS = 3
NUM_ROUNDS = 1000
MIN_CARDS_BEFORE_RESHUFFLE = 52

# Initialize Deck
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
DECK = [(rank, suit) for suit in SUITS for rank in RANKS]

# Create Shoe
shoe = DECK * NUM_DECKS
random.shuffle(shoe)

# Initialize Card Count
card_count = {card: 0 for card in RANKS} # card_count = {'2':0,'3':0,.....,'A':0}
# Function to calculate hand value
def calculate_hand_value(hand):
    value, aces = 0, 0
    for card in hand:
        rank, _ = card
        if rank in ['J', 'Q', 'K']:
            value = value + 10
        elif rank == 'A':
            aces = aces + 1
        else:
            value += int(rank)
    for _ in range(aces):
        value += 11 if value + 11 <= 21 else 1
    return value

# Update Card Count Function
def update_card_count(card):
    rank, _ = card
    card_count[rank] += 1

# Probability Calculation Function
def calculate_probabilities(num_decks_remaining, card_count):
    probabilities = {}
    total_cards = num_decks_remaining * 52 
    for card in RANKS:
        count_of_card = card_count.get(card, 0)
        probabilities[card] = count_of_card / total_cards
    return probabilities

# Decision Making Function
def make_decision(hand, dealer_card, num_decks_remaining):
    hand_value = calculate_hand_value(hand)
    probabilities = calculate_probabilities(num_decks_remaining, card_count)

    # Calculate probability of a safe hit (not exceeding 21)
    safe_hit_prob = 0
    for rank, prob in probabilities.items():
        if rank in ['J', 'Q', 'K']:
            added_value = 10
        elif rank == 'A':
            added_value = 11 if hand_value + 11 <= 21 else 1
        else:
            added_value = int(rank)
        
        if hand_value + added_value <= 21:
            safe_hit_prob += prob

    # Decision Logic
    if hand_value < 12 or (hand_value < 21 and safe_hit_prob > 0.5):
        return 1  # Hit
    else:
        return 0  # Stay

# Game Simulation and Data Collection
data = []
player_hands = [[] for _ in range(NUM_PLAYERS)]
dealer_hand = [shoe.pop()]  # Dealer's face-down card
# we will do this after the hand is over
#update_card_count(dealer_hand[0])

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
    num_decks_remaining = len(shoe) // 52
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
        
    update_card_count(dealer_hand[0])

    # Record data
    data.append({
        'num_players': NUM_PLAYERS,
        'num_decks': NUM_DECKS,
        'Player Initial Hand': calculate_hand_value(player_hand[:2]),
        'Dealer Up Card': calculate_hand_value([dealer_card]),
        'Action Taken': action,
        'Outcome': outcome
    })

# Export to CSV
df = pd.DataFrame(data)
print(df.head())
df.to_csv(f"{NUM_DECKS}_{NUM_PLAYERS}_blackjack_training_data.csv", index=False)
