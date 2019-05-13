# -*- coding: utf-8 -*-
"""
Created on Mon May 13 00:14:01 2019

@author: Iman
"""

import requests
from tqdm import tqdm
import math
import zipfile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    #CHUNK_SIZE = 32768
    CHUNK_SIZE = 1024
    total_size = 1048431506
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE),  total=math.ceil(total_size/CHUNK_SIZE) , unit='KB', unit_divisor=1024, unit_scale=True):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id = '14gDgIZIPvvUv2wjK8x1KXR8yognrgZ2X'
    destination = 'BestModelCheckpoint.zip'
    download_file_from_google_drive(file_id, destination)
    zip_ref = zipfile.ZipFile(destination, 'r')
    zip_ref.extractall('.')
    zip_ref.close()