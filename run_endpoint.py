import requests

api_endpoint = 'http://0.0.0.0/predict/0.1.1'

payloads = [
    {
    "bedrooms": 3,
    "bathrooms": 1,
    "sqft_living": 1180,
    "sqft_lot": 5650,
    "floors": 1,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "sqft_above": 1180,
    "sqft_basement": 0,
    "yr_built": 1955,
    "yr_renovated": 0,
    "zipcode": 98178,
    "latitude": 47.5112,
    "longitude": -122.257,
    "sqft_living15": 1340,
    "sqft_lot15": 5650
    },
    {
    "ID": 7129300520,
    "date": "20141013T000000",
    "bedrooms": 1,
    "bathrooms": 1,
    "sqft_living": 980,
    "sqft_lot": 3650,
    "floors": 1,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "sqft_above": 980,
    "sqft_basement": 0,
    "yr_built": 1930,
    "yr_renovated": 0,
    "zipcode": 98178,
    "latitude": 47.5112,
    "longitude": -122.257,
    "sqft_living15": 980,
    "sqft_lot15": 3650
    }
]

# Make a POST request with JSON data
for p in payloads:
    print(f"Payload details: \n {p}")
    response = requests.post(api_endpoint, json=p)

    if response.status_code == 200:
        print(f"Prediction:\n {response.json()}\n\n")
    else:
        print(f"Error: {response.status_code}")
