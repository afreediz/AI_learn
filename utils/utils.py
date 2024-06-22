import os
import requests
import json
import pickle
import pandas as pd

current_dir = os.path.dirname(__file__)

def get_data_web(url, is_json:False, save_path=None):
    data = requests.get(url).text
    if is_json:
        data = json.loads(data)
    print(data)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(str(data))
    return data

def save_pred(data, path="../predictions/", filename="pred.csv", type="csv"):
    save_path = os.path.join(current_dir, path)
    file_path = os.path.join(save_path, filename)

    if(type == "csv"):
        data.to_csv(file_path)
        print('saving to ', file_path)
        return

    with open(file_path, 'w') as f:
        f.write(str(data))

def read_pred(path="../predictions/", filename="pred.csv"):
    return open(path+filename, 'r').read()

def save_model(model, path="../models/", filename="model.pkl"):
    save_path = os.path.join(current_dir, path)
    file_path = os.path.join(save_path, filename)

    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def read_model(path="../models/", filename="model.pkl"):
    return pickle.load(open(path+filename, 'rb'))

if __name__ == "__main__":
    print("This is module")