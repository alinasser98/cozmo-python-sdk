import qrcode

suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']

for suit in suits:
    for value in values:
        img = qrcode.make(f'{value}_{suit}')
        img.save(f'{value}_{suit}.png')
