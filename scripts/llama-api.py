import requests

url = "http://192.168.200.10:11434/api/chat"

headers = {
    "Content-Type": "application/json",
    "Authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImJiMjYzNjU0LTk0NTgtNDgxZC1iN2IzLTgwNDNkOTBjZjllMiJ9.BdRhAJxbzN_tVcs_9xfd2ldJIprX-qD4KDwuyaKaGnQ"  # If required
}

payload = {
    "messages": [
        {"role": "user", "content": "Hello, who are you?"}
    ],
    "model": "deepseek-r1:14b",  # Adjust to your model
    "stream": False  # If the API supports streaming
}

response = requests.post(url, headers=headers, json=payload)

if response.ok:
    data = response.json()
    print(data)
else:
    print(f"Error {response.status_code}: {response.text}")