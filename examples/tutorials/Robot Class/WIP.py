import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

card_types = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']
special_cards = ['A', 'J', 'Q', 'K']
#need number of decks in a shoe
num_decks = 4
#need number of players at the table
num_of_players = 3
#number of simulations to run
num_of_simulations = 1000

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
 #simulate one game of blackjack, Once the cards have been dealt
   
#simulate a game of blackjack

'''
dealer_hand: 2 cards that the dealer has
player_hands: an array for the values of the cards in the player hands
curr_player_results: a numpy containing the results of the each players hand; 3 players: [1, 0, -1] w t l
dealer_cards: cards left in the shoe; if a player hits, we will remove the top card from the list
hit_stay: the decision of the player to hit or stay; 1 for hit, 0 for stay; the decision making process
card_count: a dictionary to store the cards that been romoved from the shoe
dealer_bust: did the dealer bust; a boolean to indicate if the dealer has busted

NOTE: live_action keeps track if if a player hits or stays during the hand; is a list of the actions that the players take; 1 for hit, 0 for stay
'''
def play_hand(dealer_hand, player_hands, curr_player_results, dealer_cards, hit_stay, card_count, dealer_bust):
    #check if the dealer has blackjack(so, check if the dealer has two cards and their total is is 21)
    if(len(dealer_hand)==2) and (find_total(dealer_hand)==21):
        #if the dealer has blackjack, check if the player has blackjack
        #it is a tie, otherwise the player loses
        for player in range(num_of_players):
            #update the live_action for the players, since they do not have a choice
            #live_action.append(0) # live_action is not defined
            
            #check if any player has blackjack
            if (len(player_hands[player])==2) and (find_total(player_hands[player])==21):
                #if the player has blackjack, it is a tie
                curr_player_results[0, player] = 0
            else:
                #if the player does not have blackjack, the player loses
                curr_player_results[0, player] = -1
                
    #the dealer did not have blackjack
    #let's make a decision for the player
    else:
        for player in range(num_of_players): # players is not defined
            action = 0 # action is not accessed
                
            #check to see if the player has blackjack:
            if (len(player_hands[player])==2) and (find_total(player_hands[player])==21):
                #we have a blackjack so we win.
                curr_player_results[0, player] = 1
            else:
                #the player does not have blackjack
                while (random.random() > hit_stay) and (find_total(play_hand[player]) < 12):
                    #deal a card while this is true
                    player_hands[player].append(dealer_cards.pop(0))
                    #update the dictionary to keep track of the card count
                    card_count[player_hands[player][-1]] += 1
                    
                    #i descided to hit so i need to update the action
                    action = 1
                    
                    #get the new value of the current hand and update the live_action
                    live_total.append(find_total(player_hands[player]))
                    
                    #see if the player busted:
                    if find_total(player_hands[player]) > 21:
                        #the player busted
                        curr_player_results[0, player] = -1
                        break
                #update the live_action with the players choice
                live_action.append(action)
    #return the results for each player
    return curr_player_results, dealer_cards, card_count, dealer_bust

#we still need to deal with the dealer to finish the simulation
#so, lets build the simulation
#keep track of history
dealer_card_history = []
dealer_total_history = []
player_card_history = []
player_total_history = []
outcome_history = []
player_live_total = []
player_live_action = []
dealer_bust = []

card_count_list = []
#track the characteristics of the shoe or simulation
first_game = True
prev_sim = 0
sim_number_list = []
new_sim = []
games_played_in_sim = []

for sim in range(simulations):
    #set up our card counter dictionary
    card_count = {'A':0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 'J':0, 'Q':0, 'K':0}
    
    dealer_cards = make_shoe(num_decks, card_types)
    
    while len(dealer_cards) > 20:
        #manage the game in the simulation
        