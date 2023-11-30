import random
import pandas as pd
import math

# Parameters
NUM_DECKS = 4
NUM_PLAYERS = 3
NUM_ROUNDS = 1000
MIN_CARDS_BEFORE_RESHUFFLE = 52

# Deck Creation Without Suits
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
DECK = RANKS * 4  # Four of each rank to represent four suits

# Function to create and shuffle a new shoe
def create_shuffled_shoe(num_decks):
    shoe = DECK * num_decks
    random.shuffle(shoe)
    return shoe

# Function to calculate hand value
def calculate_hand_value(hand):
    value, aces = 0, 0
    for rank in hand:
        if rank in ['J', 'Q', 'K']:
            value += 10
        elif rank == 'A':
            aces += 1
        else:
            value += int(rank)

    for _ in range(aces):
        if value + 11 <= 21:
            value += 11
        else:
            value += 1

    return value

# Update seen cards function
def update_seen_cards(rank):
    seen_cards[rank] += 1

# Probability calculation function - Based on seen cards
def calculate_probabilities(seen_cards, total_deck_count):
    probabilities = {}
    for rank in RANKS:
        unseen_count = total_deck_count[rank] - seen_cards.get(rank, 0)
        total_unseen = sum(total_deck_count.values()) - sum(seen_cards.values())
        if total_unseen > 0:
            probabilities[rank] = unseen_count / total_unseen
        else:
            probabilities[rank] = 0
    return probabilities

# Decision making function - Using estimated probabilities
def make_decision(hand, dealer_card, seen_cards, total_deck_count):
    hand_value = calculate_hand_value(hand)
    probabilities = calculate_probabilities(seen_cards, total_deck_count)
    safe_hit_prob = 0

    for rank, prob in probabilities.items():
        if rank in ['J', 'Q', 'K']:
            added_value = 10
        elif rank == 'A':
            if hand_value + 11 <= 21:
                added_value = 11
            else:
                added_value = 1
        else:
            added_value = int(rank)

        if hand_value + added_value <= 21:
            safe_hit_prob += prob

    if hand_value < 12 or (hand_value < 21 and safe_hit_prob > 0.5):
        return 1
    else:
        return 0

# Initialize seen cards and shoe
seen_cards = {card: 0 for card in RANKS}
total_deck_count = {card: NUM_DECKS * 4 for card in RANKS}
shoe = create_shuffled_shoe(NUM_DECKS)

# Data collection
data = []

# Game simulation loop
for _ in range(NUM_ROUNDS):
    if len(shoe) < MIN_CARDS_BEFORE_RESHUFFLE:
        shoe = create_shuffled_shoe(NUM_DECKS)
        seen_cards = {card: 0 for card in RANKS}

    player_hands = [[] for _ in range(NUM_PLAYERS)]
    dealer_hand = [shoe.pop()]
    update_seen_cards(dealer_hand[0])

    for player_hand in player_hands:
        card = shoe.pop()
        player_hand.append(card)
        update_seen_cards(card)

    dealer_card = shoe.pop()
    dealer_hand.append(dealer_card)
    update_seen_cards(dealer_card)

    for player_hand in player_hands:
        card = shoe.pop()
        player_hand.append(card)
        update_seen_cards(card)

    for player_hand in player_hands:
        action = make_decision(player_hand, dealer_card, seen_cards, total_deck_count)
        if action == 1:
            new_card = shoe.pop()
            player_hand.append(new_card)
            update_seen_cards(new_card)

        player_value = calculate_hand_value(player_hand)
        dealer_value = calculate_hand_value(dealer_hand)
        if dealer_value > 21 or player_value > dealer_value:
            outcome = 1
        elif player_value == dealer_value:
            outcome = 0
        else:
            outcome = -1

        current_probabilities = calculate_probabilities(seen_cards, total_deck_count)

        round_data = {
            'num_players': NUM_PLAYERS,
            'num_decks': NUM_DECKS,
            'Player Initial Hand': calculate_hand_value(player_hand[:2]),
            'Dealer Up Card': calculate_hand_value([dealer_card]),
            'Action Taken': action,
            'Outcome': outcome
        }
        for card in RANKS:
            round_data[f'Seen_{card}'] = seen_cards[card]
            round_data[f'Probability_{card}'] = current_probabilities[card]

        data.append(round_data)

# Convert to DataFrame and export to CSV
df = pd.DataFrame(data)
csv_filename = f"{NUM_DECKS}_{NUM_PLAYERS}_blackjack_training_data.csv"
df.to_csv(csv_filename, index=False)
