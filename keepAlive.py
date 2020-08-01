from PIL import Image
import requests
from io import BytesIO
import time


url="https://sih-heroku.herokuapp.com/api/classify"
while(True):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    print("on")
    time.sleep(60*50)
# img.show()