import requests
import json

def get_data_web(url, is_json:False, save_path=None):
    data = requests.get(url).text
    if is_json:
        data = json.loads(data)
    print(data)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(str(data))
    return data


get_data_web(
    url="https://gist.githubusercontent.com/heather229/7accf105b30d9122c1a5/raw/0709fba655e4a5157ffef8beab2665213d964d63/BetterLifeIndex2015.csv",
    is_json=False,
    save_path="better_life_index.csv"
    )