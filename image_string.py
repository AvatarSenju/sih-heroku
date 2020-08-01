# import base64
 
# with open("./test/abr1.jpg", "rb") as imageFile:
#     strl = base64.b64encode(imageFile.read())
#     print (strl)

# fh = open("imageToSave.png", "wb")
# fh.write(strl.decode('base64'))
# fh.close()

import base64
from PIL import Image
import io

image = open('./test/abr1.jpg', 'rb')

with open('./test/abr1.jpg', "rb") as image:
    b64string = base64.b64encode(image.read())


f = io.BytesIO(base64.b64decode(b64string))
pilimage = Image.open(f)

# image_read = image.read()
# image_64_encode = base64.encodestring(image_read)
# print(image_64_encode.decode('ASCII'))
# sl="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"
# image_64_decode = base64.decodestring(sl.encode('utf-8')) 
# image_result = open('decode.jpg', 'wb') # create a writable image and write the decoding result
# image_result.write(image_64_decode)