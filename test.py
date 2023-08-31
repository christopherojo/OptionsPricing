import requests

url = 'https://query1.finance.yahoo.com/v7/finance/options/TSLA'
headers = {
    'User-Agent': 'Your User Agent String'
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print('Request successful!')
else:
    print(f'Request failed with status code: {response.status_code}')
    print('Response content:', response.content)