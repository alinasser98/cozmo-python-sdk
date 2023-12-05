import cozmo

from cozmo.util import degrees
from PIL import Image
#NOTE: in a terminal, open python
'''
import socket
s = socket.socket()
s.connect(('10.0.1.10', 5000))
s.sendall(b'message') or s.recv(4096)

'''
def get_card_from_image(image: Image):
    from pyzbar.pyzbar import decode
    decoded_objects = decode(image)
    for obj in decoded_objects:
        return obj.data.decode('utf-8')
    return None

def cozmo_program(robot: cozmo.robot.Robot):
    hand_value = 0
    card_names = []

    for _ in range(2):
        robot.say_text("Show me one of your cards").wait_for_completed()

        latest_image = robot.world.latest_image.raw_image
        card = get_card_from_image(latest_image)

        if card:
            value, suit = card.split('_')
            if value in ['Jack', 'Queen', 'King']:
                card_value = 10
            elif value == 'Ace':
                card_value = 11 if hand_value <= 10 else 1  # Use 11 if it won't bust the hand, otherwise use 1
            else:
                card_value = int(value)

            hand_value += card_value
            card_names.append(card)

            robot.say_text(f'{value} of {suit}, hand value is {hand_value}').wait_for_completed()

    # Cozmo's decision to HIT or STAY:
    if hand_value < 17:  # This is a common rule of thumb for blackjack. Cozmo will hit if his hand value is less than 17.
        robot.say_text("HIT").wait_for_completed()
        robot.drive_straight(distance_mm(50), speed_mmps(50)).wait_for_completed()
    else:
        robot.say_text("STAY").wait_for_completed()
        robot.turn_in_place(degrees(360)).wait_for_completed()

    # Send data over network
    message = f"robobuddy;{card_names[0]};{card_names[1]}"
    send_data(message)

def send_data(message: str):
    SERVER_IP, SERVER_PORT = '10.0.1.10', 5000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((SERVER_IP, SERVER_PORT))
        s.sendall(message.encode('utf-8'))

cozmo.run_program(cozmo_program)
