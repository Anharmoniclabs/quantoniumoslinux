"""Quick test: send a chat request to the RFTMW server."""
import http.client
import json

conn = http.client.HTTPConnection("localhost", 8000, timeout=300)
body = json.dumps({
    "model": "qwen2.5-coder:3b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": False,
})
conn.request("POST", "/api/chat", body=body,
             headers={"Content-Type": "application/json"})
resp = conn.getresponse()
data = json.loads(resp.read().decode())
print(f"Status: {resp.status}")
print(f"Response: [{data['message']['content']}]")
conn.close()
