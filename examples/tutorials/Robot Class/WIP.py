import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

card_types = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']

#lets make a shoe
def make_shoe(num_decks, card_types):
    new_shoe = []
    #put the cards in the shoe
    for i in range(num_decks):
        for j in range(4):
            new_shoe.extend(card_types)
    #shuffle the cards
    random.shuffle(new_shoe)
    return new_shoe

#write a function to get the value in a hand
#pass a hand in as a list of cards and resturn the value in the hand
#count aces as 11 unless it would make the hand bust otherwise count them as 1
#face cards have a value of 10
#need to count the number of aces in the hand and determine if the value of the hand is over 21
#finalailly return the value of the hand after examining the other cards in hand
#todo: improve this component
def find_total(hand):
    face_cards = ['J', 'Q', 'K']
    aces = 0
    hand_value = 0
    #a card will either be a face card, an ace, or an integer
    #for a face card, add 10 to the hand value
    #for an integer, add the integer to the hand value
    #for an ace, count it and determine the impact on the hand value at the end
    for card in hand:
        if card == 'A':
            aces += 1
        elif card in face_cards:
            hand_value += 10
        else:
            hand_value += card
    #at this point we can determine the value of the hand with the aces
    if aces == 0:
        return hand_value
    else:
        hand_value += aces
        if hand_value + 10 <= 21:
            return (hand_value + 10)
        else:
            return hand_value
    