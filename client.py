import requests

url = "http://127.0.0.1:8000/batch_inference"

# Prepare the image files
# files = [
#     ('files', ('apple.jpg', open('apple.jpg', 'rb'), 'image/jpeg')),
#     ('files', ('banana.jpg', open('banana.jpg', 'rb'), 'image/jpeg')),
# ] * 10

files = []
for _ in range(1):  # Send each file 10 times properly
    files.append(('files', ('apple.jpg', open('apple.jpg', 'rb'), 'image/jpeg')))
    files.append(('files', ('banana.jpg', open('banana.jpg', 'rb'), 'image/jpeg')))


# Send the request
response = requests.post(url, files=files)

# Print the response
print(response.json())