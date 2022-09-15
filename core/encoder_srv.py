"""
Encoder Service
"""
import os
import requests

ENCODER_SRV_ENDPOINT = os.environ.get("ENCODER_SRV_ENDPOINT")
assert isinstance(ENCODER_SRV_ENDPOINT, str)


def encode(data, encoder):
    """Returns the encoded version of the string using from the API.

    Args
    data (str): the data to be encoded.
    encoder: Used to Specify the encoder to use.

    Returns:
         The encoded data.
    """
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
