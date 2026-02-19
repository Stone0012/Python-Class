import requests



SERVER_URL = "http://lambda2.uncw.edu:11434/api/generate"
payload = {
    "model": 'llama3.2',
    "prompt": "Write a poem about dogs",
    "stream": False
}

response = requests.post(SERVER_URL, json=payload)
response_json = response.json()  # Convert response to dictionary
print(response_json.get("response", "Key 'response' not found"))