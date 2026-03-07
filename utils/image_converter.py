import base64
import io
from PIL import Image

# Converts a base64 string (with or without header) into a PIL Image.
def decode_base64_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
 
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))

        return img.convert('RGB')
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None
