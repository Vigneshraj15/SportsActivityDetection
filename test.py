import requests

url = "http://127.0.0.1:5000/predict"
image_path = r"E:\WINTER SEMESTER 24-45\DIGITAL IMAGE PROCESSING\archive\test\Image_227.jpg"

with open(image_path, "rb") as img:
    files = {"file": img}
    response = requests.post(url, files=files)

print("Raw Response:", response.text)  # Debugging
try:
    print("JSON Response:", response.json())  
except requests.exceptions.JSONDecodeError:
    print("Error: Response is not valid JSON!")
