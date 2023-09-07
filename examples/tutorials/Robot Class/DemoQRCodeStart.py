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

myText = 'Let us create a QR code'
img = qrcode.make(myText)
img.save('DemoQRCode.png')