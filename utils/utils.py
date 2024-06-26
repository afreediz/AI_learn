import os
import tarfile
import urllib
import requests
import json

current_dir = os.path.dirname(os.path.abspath(__file__))

def get_data_web(url, is_json:False, save_path=None):
    data = requests.get(url).text
    if is_json:
        data = json.loads(data)
    print(data)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(str(data))
    return data

base_path = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
housing_url = "datasets/housing/housing.tgz"
save_path = os.path.join(current_dir, "../datasets/housing/")


def fetch_housing_data(housing_url):
    os.makedirs(save_path, exist_ok=True)
    tgz_path = os.path.join(save_path, "housing.tgz")
    urllib.request.urlretrieve(base_path+housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(save_path)
    housing_tgz.close()
    os.remove(tgz_path)




# file retrieved is in csv format, no need of unnzip.
fetch_housing_data(housing_url)