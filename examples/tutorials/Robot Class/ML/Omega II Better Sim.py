#neural network for blackjack

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout

#let's set up our parameters
num_decks = 6
players = 6
deck_penetration = 0.75  # 75% as an example


#first, let's make a shoe
def make_shoe(num_decks, card_types):
    new_shoe = []
    for i in range(num_decks):
        for j in range(4):
            new_shoe.extend(card_types)
    random.shuffle(new_shoe)
    return new_shoe

def decks_remaining(cards):
    return len(cards) / 52.0

#next, we need a function to add up the value of the cards in a hand.  Aces can be 1 or 11, and we need to allow for both
#a hand consists of a list of cards, and this function returns a list of the possible values of the total, depending on the
#number of aces.  Because a shoe has several decks, we need to allow for the possibility of lots of aces.  If there is one
#ace, the ace can be either 1 or 11.  two aces can add up to 2 or 12 (NOTE: only one ace in a hand can count as 11)
#in general, k aces can add up to either k or k + 10, but we only care about an ace being 11 if it doesn't make the player go bust
#as a naive approach, we will just consider the value of the ace to be the one that yields the highest hand value...this might change
#in your decision-making strategy, especially if counting it as an 11 results in a bust, but counting it as a 1 keeps a player alive
def find_total(hand):
    face_cards = ['K', 'Q', 'J']
    aces = 0
    total = 0
    #a card will either be an integer in [2, 10] or a face card, or an ace
    for card in hand:
        if card == 'A':
            aces = aces + 1
        elif card in face_cards:
            total = total + 10
        else:
            total = total + card
        #at this point, we have the total of the hand, excluding any aces. 
    if aces == 0:
        return total
    else:
        #count all of the aces as 1, and return the highest of the possible total values, in ascending order; this is one place
        #where our approach could be improved
        ace_hand = [total + aces]
        ace_total = total + aces + 10
        if  ace_total < 22:
            ace_hand.append(ace_total)
        return max(ace_hand)

def prob_based_on_omega_ii(card_count, dealer_cards):
        # Implementing Omega II card counting strategy
    omega_ii_count = card_count[2] + card_count[3] + \
                     2*card_count[4] + 2*card_count[5] + 2*card_count[6] + \
                     card_count[7] - card_count[8] - 2*card_count[9] - \
                     2*card_count[10] - 2*card_count['J'] - 2*card_count['Q'] - \
                     2*card_count['K']
    # Calculate true count
    true_count = omega_ii_count / decks_remaining(dealer_cards)

    #use true_count to adjust your hit probabilities
    if true_count > 0:
        return 0.75
    else:
        return 0.25
#ONE game, once the cards have been dealt...we will use this function to determine the player strategy
#dealer_hand: 2 cards the dealer has
#player_hands: the cards that the players have
#curr_player_results: a list containing the result of each player's hand for this round; if there are three players, it might be [1, -1, 1]
#dealer_cards: the cards left in the shoe; the shoe with the cards that have been dealt to the players for hitting will have been removed
#hit_stay: is used to determine if a player hits or stays...you'll probably modify this in your own decision-making process
#card_count: a dictionary to store the counts of the various card values that have been seen, for future card
#counting in influencing our decision making and training data
# def play_hand(dealer_hand, player_hands, curr_player_results, dealer_cards, hit_stay, card_count, dealer_bust, players):
    """
    Simulates a single hand of blackjack for the given number of players and returns the results for each player.
    
    Parameters:
    dealer_hand (list): A list of the dealer's cards.
    player_hands (list): A list of lists, where each inner list contains the cards for a player.
    curr_player_results (numpy array): A 1xN numpy array, where N is the number of players, that will be updated with the results of the current hand for each player.
    dealer_cards (list): A list of the remaining cards in the deck.
    hit_stay (float): The probability threshold for a player to hit or stay, based on the Hi-Lo method.
    card_count (dictionary): A dictionary that keeps track of the count of each card in the deck.
    dealer_bust (list): A list that keeps track of whether the dealer has gone bust in each hand.
    players (int): The number of players in the game.
    
    Returns:
    curr_player_results (numpy array): A 1xN numpy array, where N is the number of players, that contains the results of the current hand for each player.
    """
def play_hand(dealer_hand, player_hands, curr_player_results, dealer_cards, hit_stay, card_count, dealer_bust, players):
    
    #first, check if the dealer has blackjack.  that can only happen if the dealer has a total of 21, logically, and 
    #the game will be over before it really gets started...the players cannot hit
    if (len(dealer_hand) == 2) and (find_total(dealer_hand) == 21):
        for player in range(players):
            #update live_action for the players, since they don't have a choice
            live_action.append(0)
            
            #check if any of the players also have blackjack, if so, they tie, and if not, they lose
            if (len(player_hands[player]) == 2) and (find_total(player_hands[player]) == 21):
                curr_player_results[0, player] = 0
            else:
                curr_player_results[0, player] = -1    
    
    #now each player can make their decisions...first, they should check if they have blackjack
    #for this player strategy, the decision to hit or stay is random if the total value is less than 12...
    #so it is somewhat unrelated to the cards they actually have been dealt (and is conservative), and ignores the card 
    #that the dealer has.  We will use this strategy to generate training data for a neural network.  
    #your job will be to improve this strategy, incorporate the dealer's revealed card, train a new neural
    #network based on that simulated data, and then compare the results of your neural network to the baseline
    #model generated from this training data.
    else:
        for player in range(players):
            #the default is that they do not hit
            action = 0
            
            #check for blackjack so that the player wins
            if (len(player_hands[player]) == 2) and (find_total(player_hands[player]) == 21):
                curr_player_results[0, player] = 1
            else:
                # hit based on the Hi_Lo method given in the definition of the funciton prob_base_on_cc in line 58
                probability = prob_based_on_omega_ii(card_count, dealer_cards)
                while (probability > hit_stay) and (find_total(player_hands[player]) < 17):
                    #deal a card
                    player_hands[player].append(dealer_cards.pop(0))
                    
                    #update our dictionary to include the new card
                    card_count[player_hands[player][-1]] += 1
                    
                    #note that the player decided to hit
                    action = 1
                    
                    #get the new value of the current hand regardless of if they bust or are still in the game
                    #we will track the value of the hand during play...it was initially set up in the section below,
                    #and we are just updating it if the player decides to hit, so that it changes
                    live_total.append(find_total(player_hands[player]))                      
                        
                    #if the player goes bust, we need to stop this nonsense and enter the loss...
                    #we will record their hand value outside of the while loop once we know the player is done
                    if find_total(player_hands[player]) > 21:
                        curr_player_results[0, player] = -1
                        break  
            #update live_action to reflect the player's choice
            live_action.append(action)
                    
    #next, the dealer takes their turn based on the rules
    #first, the dealer will turn over their card, so we can count it and update our dictionary; this is the FIRST card they were dealt
    card_count[dealer_hand[0]] += 1
    
    while find_total(dealer_hand) < 17:
        #the dealer takes a card
        dealer_hand.append(dealer_cards.pop(0))    
        
        #update our dictionary for counting cards
        card_count[dealer_hand[-1]] += 1
    
    
    #this round is now complete, so we can determine the outcome...first, determine if the dealer went bust
    if  find_total(dealer_hand) > 21:
        
        #the dealer went bust, so we can append that to our tracking of when the dealer goes bust
        #we'll have to track the player outcomes differently, because if the dealer goes bust, a player
        #doesn't necessarily win or lose
        dealer_bust.append(1)
        
        #every player that has not busted wins
        for player in range(players):
            if curr_player_results[0, player] != -1:
                curr_player_results[0, player] = 1
    else:
        #the dealer did not bust
        dealer_bust.append(0)
        
        #check if a player has a higher hand value than the dealer...if so, they win, and if not, they lose
        #ties result in a 0; for our neural network, we may want to lump ties with wins if we want a binary outcome
        for player in range(players):
            if find_total(player_hands[player]) > find_total(dealer_hand):
                if find_total(player_hands[player]) < 22:
                    curr_player_results[0, player] = 1
            elif find_total(player_hands[player]) == find_total(dealer_hand):
                curr_player_results[0, player] = 0
            else:
                curr_player_results[0, player] = -1    
    
    #the hand is now complete, so we can return the results
    #we will return the results for each player
    return curr_player_results, dealer_cards, card_count, dealer_bust
                 

#now we can run some simulations

card_types = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K'] 
special_cards = [10, 'J', 'Q', 'K'] 

#set some parameters for the number of simulations (each simulation involves going through a shoe)
simulations = 1000


#let's keep track of each round (dealer and player hands as well as the outcome) to analyze this data
#there is no need to break this up by simulation, since what we want to analyze are the games, regardless
#of which simulation it is in...but we will be able to track that information through the sim_number_list
#if we wanted to analyze our data across simulations.

#we want the cards that the dealer was dealt throughout the simulaton
dealer_card_history = []
dealer_total_history = []

#we want all of the cards dealt to each player for each of the games in the simulation
player_card_history = []

#we want the player's outcome for each of the games in the simulation
outcome_history = []

#we want the hand values tracked for each of the games in the simulation
player_live_total = []

#we want to know whether the player hit during each of the games in the simulation
player_live_action = []

#we want to know if the dealer went bust in each of the games in the simulation
dealer_bust = []

#we need to keep track of our card counter throughout the simulation
card_count_list = []

#we can track characteristics related to the shoe or simulation, as noted above:
first_game = True
prev_sim = 0
sim_number_list = []
new_sim = []
games_played_in_sim = []


#let's run our simulations

for sim in range(simulations):
    #randomly assigning the num decks and num of players
    #num_decks = random.randint(6, 10)
    #players = random.randint(6, 14)

    #we aren't recording our data by simulation, but we could if we changed our minds
    #dealer_card_history_sim = []
    #player_card_history_sim = []
    #outcome_history_sim = []    
    
    games_played = 0
    
    #create the shoe
    dealer_cards = make_shoe(num_decks, card_types)
    
    #for each simulation, create a dictionary to keep track of the cards in the shoe, initially set to 0 for all cards
    card_count = {'A': 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 1, 8: 0, 9: -1, 10: -2, 'J': -2, 'Q': -2, 'K': -2}    
    
    #play until the shoe is almost empty...we can change this to be a function of the number of decks
    #in a shoe, but we won't start a game if there are fewer than 20 cards in a shoe...if we limit
    #the number of players to 4 (plus the dealer), then we'll need at least 10 cards for the game, and
    #we'll have enough cards for everyone to take 2...here's where the card counting could work to
    #a player's advantage
    cards_before_reshuffle = (1 - deck_penetration) * num_decks * 52

    while len(dealer_cards) > cards_before_reshuffle:
        
        #here's how we will manage each game in the simulation:
        
        #keep track of the outcome of the players hand after the game: it will be 1, 0, -1
        curr_player_results = np.zeros((1, players))
        
        #create the lists for the dealer and player hands
        dealer_hand = []
        player_hands = [ [] for player in range(players)]
        live_total = []
        live_action = []
        
        #deal the FIRST card to all players and update our card counting dictionary
        for player, hand in enumerate(player_hands):
            player_hands[player].append(dealer_cards.pop(0))
            card_count[player_hands[player][-1]] += 1
            
        #dealer gets a card, and the card counting dictionary is NOT updated
        dealer_hand.append(dealer_cards.pop(0))
        #card_count[dealer_hand[-1]] += 1
        
        #deal the SECOND card to all players and update our card counting dictionary
        for player, hand in enumerate(player_hands):
            player_hands[player].append(dealer_cards.pop(0))
            card_count[player_hands[player][-1]] += 1
            
        #the dealer gets a card, and our card counter will be updated with the card that is showing
        dealer_hand.append(dealer_cards.pop(0))
        card_count[dealer_hand[-1]] += 1
        
        #record the player's live total after cards are dealt...if a player hits, we will update this information
        live_total.append(find_total(player_hands[player]))
        
        #flip a fair coin to determine if the player hits or stays...we can create a bias if we want to 
        #make this more sophisticated
        hit_stay = 0.5
        
        curr_player_results, dealer_cards, card_count, dealer_bust = play_hand(dealer_hand, player_hands, curr_player_results, dealer_cards, hit_stay, card_count, dealer_bust, players)
        
        #track the outcome of the hand
        #we want to know the dealer's card that is showing and their final total
        dealer_card_history.append(dealer_hand[1])
        dealer_total_history.append(find_total(dealer_hand))
        
        #this is the result of the hand for the players
        player_card_history.append(player_hands)
        
        #we want the outcome of each hand for each player
        outcome_history.append(list(curr_player_results[0]))
        
        #we want the evolution of each player's hand in a game, as well as whether they hit or not (this is 1 if the player ever hit)
        player_live_total.append(live_total)
        player_live_action.append(live_action)
        
        if sim != prev_sim:
            new_sim.append(1)
        else:
            new_sim.append(0)
            
        if first_game == True:
            first_game = False
        else:
            games_played += 1
        
        sim_number_list.append(sim)
        games_played_in_sim.append(games_played)
        card_count_list.append(card_count.copy())
        prev_sim = sim
        
           
#create the dataframe for analysis.  My model will have the following features:
#the dealer's second card is the one that is face up...
#the player's initial hand value
#whether the player hit or not
#whether the dealer went bust or not
#the dealer's total hand value

#the outcome (not the attribute that will be determined to be the label in our model): win or lose

model_df = pd.DataFrame()
model_df['dealer_card'] = dealer_card_history
model_df['dealer_value'] = dealer_total_history

#get initial hand values for all of the players and put them in the dataframe
dealt_hand_values = []
for i in range(len(player_card_history)):
    hand_list = []
    hands = player_card_history[i]
    for j in range(len(hands)):
        hand_list.append(find_total(hands[j][0:2]))
    dealt_hand_values.append(hand_list.copy())
    
model_df['player_initial_value'] = dealt_hand_values

#get the action for each player for each game
model_df['hit'] = player_live_action
model_df['dealer_bust'] = dealer_bust

#did the players win or lose? we will include a tie as a win for binary classification purposes
model_df['results'] = outcome_history

#now let's think about our model...it will determine if we should have hit or should have stayed.  There are many ways
#to think about the logic here.  We aren't going to focus on just winning or losing, but the decision 
#that was made and the resulting outcome.  The idea is that we can give the card information and specify the
#action taken (hit or stay) and determine what we should have done.

#so, if the player hit and won/tied, then hitting was the correct decision
#if the player hit and lost, then hitting was the wrong decision (they should have stayed)
#if the player stayed and won/tied, then staying was the correct decision
#if the player stayed and lost, then staying was the incorrect decision (they should have hit)
#let's create the outcome of interest, which we will call Y.  If the player hit an it was the correct
#decision, Y = 1, and if the player made the incorrect decision, Y = 0 (they should not have hit)
#if the player stayed and it was the incorrect decision, they should have hit, so Y = 1
#if the player staybed and it was the correct decision, then Y = 0, and if it was the incorrect decision
#then Y = 1

#we can think about what to do about situations where there is no decision (the dealer gets blackjack)
#will we remove that data, and only give the neural net training data that involves being able
#to make a decision?  That is something for you to consider!  There are flaws in this reasoning, however.
#it is possible for there to not be a "right decision": hit or stay results in a loss, for example.
#or hit or stay results in a win.  So, we have to realize that our logic is not atually complete.

#it would be nice to have a formula to map (action, outcome) to the reommended player action: hit: Y = 1, stay Y = 0 
#decisions: (0, 0) --> Y = 0 (stay, tie --> stay)   (0, 1) --> Y = 0 (stay, win --> stay)  (0, -1) --> Y = 1 (stay, lose --> hit)
#decisions: (1, 0) --> Y = 1 (hit, tie --> hit)     (1, 1) --> Y = 1 (hit, win --> hit)    (1, -1) --> Y = 0 (hit, lose --> stay)
#i decided to just write some nested conditional statements for simpliity
decision_evaluation = []
for i in range(len(player_live_action)):
    #get the two arrays for the action taken (hit or stay) and the outcome (1: win, 0: tie, -1: lose)
    action_list = player_live_action[i]
    outcome_list = outcome_history[i]
    interest_list = []
    for k in range(len(action_list)):
        #action = 0 means they stayed
        if action_list[k] == 0:
            #stay and lose --> should have hit
            if outcome_list[k] == -1:
                value = 1
            else:
                #stay and win or tie -->should stay
                value = 0
        else:
            #action = 1 means they hit
            if outcome_list[k] == -1:
                #hit and lose --> should stay
                value = 0
            else:
                #hit and win or tie --> should hit
                value = 1
        interest_list.append(value)
        
        
    #decision_evaluation will be the result that our model will predict (should we have hit? Y = 1.  Should we have stayed? Y = 0)
    decision_evaluation.append(interest_list.copy())

#this attribute ('outcome') will be our label in our model...the parameter that we want our model to determine
model_df['outcome'] = decision_evaluation

#our data is now complete, but it's not in the format that we want.  Most of the work involved in training
#an ML model involves generating, cleaning, and transforming the data.  One that's done, the rest is straightforward.
#the last step is to split up the information so that we can focus on each player and each hand separately
#we urrently have arrays for the player_initial_value, hit, results, and outcome attributes

#NOTE: For this proess, I'm intentionally stepping through each logial step in the process, rather than just
#writing a minimal script to accomplish this task.  bite-size pieces help ensure that the logic is correct. 
#this is not the only way to accomplish this task.  But, hopefully you are reading each section to understand
#what i'm doing...especially if you don't have much experience working with pandas.

#we'll first expand all of the arrays in our dataframe.  In this process, a prefix will be added so
#that we can keep track of what each feature means.  I'm creating new dataframes for each of these expansions
#the player_initial_value, hit, results, and outcome...these expansions are adding fields, NOT rows. so the index
#values should not change...and that's what we'll be able to merge on.  put some print statements in to follow these
#steps.
init_hand_df = pd.DataFrame(model_df['player_initial_value'].tolist()).fillna('').add_prefix('init_hand_p')
hit_df = pd.DataFrame(model_df['hit'].tolist()).fillna('').add_prefix('hit_p')
results_df = pd.DataFrame(model_df['results'].tolist()).fillna('').add_prefix('result_p')
outcome_df = pd.DataFrame(model_df['outcome'].tolist()).fillna('').add_prefix('outcome_p')

num_of_player = [players] * len(decision_evaluation)
num_of_decks = [num_decks] * len(decision_evaluation)
#now i have to merge these new dataframes to the original model_df.  again, some print statements (using info or describe
#on a dataframe is probably helpful here).
step1 = pd.merge(model_df, init_hand_df, left_index=True, right_index=True)
step2 = pd.merge(step1, hit_df, left_index=True, right_index=True)
step3 = pd.merge(step2, results_df, left_index=True, right_index=True)
step4 = pd.merge(step3, outcome_df, left_index=True, right_index=True)

# Drop the original arrays with player information
step5 = step4.drop(['player_initial_value', 'hit', 'results', 'outcome'], axis=1)

# Create attribute names for the columns
attribute_names = ['num_of_player', 'num_of_decks', 'dealer_card', 'dealer_value', 'dealer_bust', 'init_hand', 'hit', 'result', 'outcome']

# Create a list to store DataFrames for each player
player_dataframes = []

for index in range(players):
    att1 = 'init_hand_p' + str(index)
    att2 = 'hit_p' + str(index)
    att3 = 'result_p' + str(index)
    att4 = 'outcome_p' + str(index)
    
    # Extract the data for the current player
    player_df = step5[['dealer_card', 'dealer_value', 'dealer_bust', att1, att2, att3, att4]]
    
    # Rename the columns
    player_df.columns = attribute_names[2:]  # Skip 'num_of_player' and 'num_of_decks'
    
    # Add 'num_of_player' and 'num_of_decks' columns to the beginning
    player_df.insert(0, 'num_of_player', num_of_player[index])
    player_df.insert(1, 'num_of_decks', num_of_decks[index])
    
    # Append the player's DataFrame to the list
    player_dataframes.append(player_df)

# Concatenate all player DataFrames
final_df = pd.concat(player_dataframes, ignore_index=True)

# Write the data to a CSV file
final_df.to_csv('OmegaII_6_6.csv', index=False)

#NOTE: Now we are ready to train/test the model, using the data in this csv.  You should think about automating this process
#so that you can easily generate csv files with simulation data for different scenarios (varying the number of decks in a shoe
#or the number of players.