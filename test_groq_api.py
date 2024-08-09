import requests

# Hard-code the Groq API key here
GROQ_API_KEY = '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'

def test_groq_api():
    prompt = "Hello, how are you?"
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'prompt': prompt,
        'max_tokens': 50
    }
    try:
        response = requests.post('https://api.groq.com/v1/completions', headers=headers, json=data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("API Response:", response.json().get('choices')[0].get('text').strip())
        else:
            print("Error: Could not get a response from the API.")
            print("Response Content:", response.text)
    except Exception as e:
        print(f"An exception occurred: {e}")

if __name__ == "__main__":
    test_groq_api()
