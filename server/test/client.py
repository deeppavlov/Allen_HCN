import requests
from server.src.app import MESSAGE_URI

BASE_URL = "http://localhost:5000"
MASSAGE_URL = BASE_URL + MESSAGE_URI
post_data = {'user_message': 'test message'}


def fetch_response(url, **kwargs):
    print("Starting request...")
    response = requests.post(url, **kwargs)
    response.raise_for_status()
    print(response.text)


fetch_response(MASSAGE_URL, json=post_data)
