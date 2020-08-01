# import base64
 
# with open("./test/abr1.jpg", "rb") as imageFile:
#     strl = base64.b64encode(imageFile.read())
#     print (strl)

# fh = open("imageToSave.png", "wb")
# fh.write(strl.decode('base64'))
# fh.close()

import base64
image = open('./test/abr1.jpg', 'rb')
image_read = image.read()
image_64_encode = base64.encodestring(image_read)
print(image_64_encode.decode('utf-8'))
image_64_decode = base64.decodestring(image_64_encode) 
image_result = open('decode.jpg', 'wb') # create a writable image and write the decoding result
image_result.write(image_64_decode)