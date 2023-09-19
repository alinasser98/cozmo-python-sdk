import cozmo
import socket
from socket import error as socket_error
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
import qrcode
# Generating QR codes

# suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
# values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']

# for suit in suits:
#     for value in values:
#         img = qrcode.make(f'{value}_{suit}')
#         img.save(f'{value}_{suit}.png')

# Constants for card interpretation
COZMO_NAME = "Ali Nasser"

def card_value(value):
    if value in ["Jack", "Queen", "King"]:
        return 10
    elif value == "Ace":
        return 11
    else:
        return int(value)

def cozmo_program(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    total_value = 0
    cards_detected = []

    # Establish connection
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket_error as msg:
        robot.say_text("socket failed" + msg).wait_for_completed()
    ip = "10.0.1.10"
    port = 5000
    
    try:
        s.connect((ip, port))
    except socket_error as msg:
        robot.say_text("socket failed to bind").wait_for_completed()
    cont = True
    
    robot.say_text("ready").wait_for_completed()    

    while cont:
        card = None
        image = robot.world.latest_image.raw_image
        image = image.convert('L')
        decoded = decode(image, symbols=[ZBarSymbol.QRCODE])

        if len(decoded) > 0:
            codeData = decoded[0]  # use decoded instead of decodedImage
            myData = codeData.data
            myString = myData.decode('ASCII')
            print(myString)
            card = myString
        else:
            print('I could not decode the data')
        
        if card:
            if card not in cards_detected:
                cards_detected.append(card)
                value, suit = card.split("_")
                card_val = card_value(value)
                total_value += card_val

                if value == "Ace" and total_value > 21:
                    total_value -= 10

                robot.say_text(f"{value} of {suit}, hand value is {total_value}").wait_for_completed()

                if len(cards_detected) >=2:
                    if total_value < 17:
                        robot.say_text("HIT").wait_for_completed()
                        # Move lift up
                        robot.set_lift_height(1).wait_for_completed()
                        # Move lift down
                        robot.set_lift_height(0).wait_for_completed()
                    else:
                        robot.say_text("STAY").wait_for_completed()
                        # Spin around
                        robot.turn_in_place(cozmo.util.degrees(360)).wait_for_completed()

                    # Send card details to server
                    message = f"{COZMO_NAME};{''.join(cards_detected[0])};{''.join(cards_detected[1])}"
                    print(message)
                    s.sendall(message.encode('utf-8'))
                    break

    s.close()

cozmo.run_program(cozmo_program,True,force_viewer_on_top=True)
