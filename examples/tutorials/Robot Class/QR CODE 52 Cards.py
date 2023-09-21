# Import required libraries and modules
import cozmo
import socket
from socket import error as socket_error
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
import qrcode

# For each card, it creates a QR code image and saves it.
# 
# suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
# values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
# for suit in suits:
#     for value in values:
#         img = qrcode.make(f'{value}_{suit}')
#         img.save(f'{value}_{suit}.png')

# Define Cozmo's name
COZMO_NAME = "Ali Nasser"

# Function to convert card face values to their numerical equivalents
def card_value(value):
    if value in ["Jack", "Queen", "King"]:
        return 10
    elif value == "Ace":
        return 11
    else:
        return int(value)

# Main program to be executed on the Cozmo robot
def cozmo_program(robot: cozmo.robot.Robot):
    # Enable the robot's camera stream
    robot.camera.image_stream_enabled = True
    total_value = 0  # Total value of detected cards
    cards_detected = []  # List to keep track of detected cards

    # Initialize socket connection
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket_error as msg:
        robot.say_text("socket failed" + msg).wait_for_completed()
    ip = "10.0.1.10"
    port = 5000
    
    # Attempt to connect to the specified IP and port
    try:
        s.connect((ip, port))
    except socket_error as msg:
        robot.say_text("socket failed to bind").wait_for_completed()

    robot.say_text("ready").wait_for_completed()

    cont = True
    while cont:
        card = None
        # Fetch the latest image from the robot's camera
        image = robot.world.latest_image.raw_image
        image = image.convert('L')
        decoded = decode(image, symbols=[ZBarSymbol.QRCODE])  # Decode QR codes from the image

        if len(decoded) > 0:
            codeData = decoded[0]
            myData = codeData.data
            myString = myData.decode('ASCII')
            print(myString)
            card = myString
        else:
            print('I could not decode the data')

        if card and card not in cards_detected:
            cards_detected.append(card)  # Add card to detected list
            value, suit = card.split("_")  # Extract card value and suit
            card_val = card_value(value)  # Convert card value to numerical equivalent
            total_value += card_val  # Update total hand value

            # Handle Ace values when total exceeds 21
            if value == "Ace" and total_value > 21:
                total_value -= 10

            robot.say_text(f"{value} of {suit}, hand value is {total_value}").wait_for_completed()

            # After detecting at least two cards
            if len(cards_detected) >= 2:
                # Decision to HIT or STAY based on total hand value
                if total_value < 17:
                    robot.say_text("HIT").wait_for_completed()
                    robot.set_lift_height(1).wait_for_completed()  # Move lift up
                    robot.set_lift_height(0).wait_for_completed()  # Move lift down
                else:
                    robot.say_text("STAY").wait_for_completed()
                    robot.turn_in_place(cozmo.util.degrees(360)).wait_for_completed()  # Spin around

                # Send card details to server
                message = f"{COZMO_NAME};{''.join(cards_detected[0])};{''.join(cards_detected[1])}"
                print(message)
                s.sendall(message.encode('utf-8'))
                break  # Exit the loop

    s.close()  # Close the socket connection

# Start the Cozmo program
cozmo.run_program(cozmo_program,True,force_viewer_on_top=True)
