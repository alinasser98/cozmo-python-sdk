#import the QRCode creater library: https://pypi.org/project/qrcode/
#pip install qrcode...this one is fine
import qrcode

#pip install pyqrcode...here is a different option
import pyqrcode

#import the QRCode reader libraries: https://pypi.org/project/pyzbar/
#pip install pyzbar
#pip install PIL (this you probably already have)

from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
from PIL import Image

myText = 'Let us create a QR code again so that you know this is real'
img = qrcode.make(myText)
img.save('DemoQRCode.png')

# Reading or Decoding a QR code
decoded = decode(Image.open('DemoQRCode.png'), symbols=[ZBarSymbol.QRCODE])
print(decoded)  # print decoded instead of decodedImage
if len(decoded) > 0:
    codeData = decoded[0]  # use decoded instead of decodedImage
    myData = codeData.data
    myString = myData.decode('ASCII')
    print(myString)
else:
    print('I could not decode the data')
    
    
'''Network stuff:
import socket
s = socket.socket()
s.connect(('10.0.1.10', 5000))
s.sendall(b'message')
    or 
s.recv(4096)'''