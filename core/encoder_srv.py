import os
import requests

ENCODER_SRV_ENDPOINT = os.environ.get("ENCODER_SRV_ENDPOINT")
assert isinstance(ENCODER_SRV_ENDPOINT, str)


def encode(data, encoder):
    url = f"{ENCODER_SRV_ENDPOINT}/encode"
    payload = {"data": data, "encoder": encoder}
    try:
        response = requests.post(url, json=payload)
    except Exception as e:
        raise e
    if response.status_code != 200:
        print(data)
        raise Exception(response.text)
    return response.json().get("encoded")
