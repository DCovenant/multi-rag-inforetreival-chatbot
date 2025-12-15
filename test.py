import requests
import json
import sys

# Extract everything after "python3 test.py" as the prompt
PROMPT = " ".join(sys.argv[1:])

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.1:8b", "prompt": PROMPT}
)

# Parse the response as JSON and concatenate the 'response' fields
responses = resp.text.splitlines()
fluid_text = "".join([json.loads(line).get("response", "") for line in responses if "response" in line])

print(fluid_text)

