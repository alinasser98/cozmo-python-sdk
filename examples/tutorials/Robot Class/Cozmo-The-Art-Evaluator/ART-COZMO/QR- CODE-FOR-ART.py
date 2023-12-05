import qrcode
import os

# List of art names
art_names = [
    "Cheap AI art mountain lake trees",
    "Cheap AI art street lights and trees",
    "Edvard Munch The Scream 1893",
    "Georges Seurat A Sunday Afternoon on the Island of La Grande Jatte 1884–1886",
    "Georgia O’Keeffe Red Canna 1924",
    "Johanne Vermeer The Girl With a Pearl Earring 1632-1675",
    "René Magritte The Son of Man 1964"
]

# Relative path for QR code storage
qr_code_dir = 'examples/tutorials/Robot Class/Cozmo-The-Art-Evaluator/QR-CODES-FOR-ART'
os.makedirs(qr_code_dir, exist_ok=True)

# Function to create and save QR code
def create_qr_code(text, file_path):
    img = qrcode.make(text)
    img.save(file_path)

# Create and save QR codes for each art piece
for art_name in art_names:
    file_name = art_name.replace(' ', '_').replace(',', '') + '.png'
    file_path = os.path.join(qr_code_dir, file_name)
    create_qr_code(art_name, file_path)
