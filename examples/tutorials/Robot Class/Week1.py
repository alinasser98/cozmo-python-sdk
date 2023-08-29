def getHawkID():
    myID = 'nssr'  # Replace 'YOURHAWKID' with your actual Hawk ID
    return [myID]

def parseMessage(message):
    parts = message.split(';')  # Split the message using the semicolon delimiter
    
    # Extract each part and convert x and y to integers
    name = Cozmo_768855[0]
    char1 = parts[1]
    char2 = parts[2]
    x = int(parts[3])
    y = int(parts[4])
    
    return [name, char1, char2, x, y]

# Test the function
print(parseMessage('nssr;F;L;200;300'))