import glob
import json

import numpy as np
import pyxis as px
import pyxis.torch as pxt

from utils import process_xrd_data, read_brml, read_txt

# import matplotlib.pyplot as plt


# for file in glob.glob("data/*/*"):
#     print(file)
#     if file.endswith(".brml"):
#         data = read_brml(file)
#     elif file.endswith(".txt") or file.endswith(".csv"):
#         data = read_txt(file)
#     data = process_xrd_data(data)

#     print(data.head())

#     print(data['Intensity'].min(), data['Intensity'].max())
#     print(data['TwoTheta'].min(), data['TwoTheta'].max())

ROUTES = [
    "U3O8ADU",
    "U3O8AUC",
    "U3O8MDU",
    "U3O8SDU",
    "U3O8UO4",
    "UO2AUCd",
    "UO2AUCi",
    "UO2MDU",
    "UO2SDU",
    "UO2UO4",
    "UO3AUC",
    "UO3MDU",
    "UO3SDU",
    "UO3UO4",
]

STARTMAT = [
    "U3O8",
    "UO2",
    "UO3",
]

startmat_map = {
    "U3O8ADU": "U3O8",
    "U3O8AUC": "U3O8",
    "U3O8MDU": "U3O8",
    "U3O8SDU": "U3O8",
    "U3O8UO4": "U3O8",
    "UO2AUCd": "UO2",
    "UO2AUCi": "UO2",
    "UO2MDU": "UO2",
    "UO2SDU": "UO2",
    "UO2UO4": "UO2",
    "UO3AUC": "UO3",
    "UO3MDU": "UO3",
    "UO3SDU": "UO3",
    "UO3UO4": "UO3",
}


## make a json with files and their route and starting material
fullroutes = []
startingmat = []
for route in ROUTES:
    files = glob.glob(f"data/{route}/*")
    for file in files:
        fullroutes.append({"label": route, "file": file})
        startingmat.append({"label": startmat_map[route], "file": file})


# write json files
with open("data/fullroutes.json", "w", encoding="utf-8") as f:
    json.dump(fullroutes, f, indent=4)

with open("data/startingmat.json", "w", encoding="utf-8") as f:
    json.dump(startingmat, f, indent=4)

# make fullroutes lmdb
with px.Writer(dirpath="./data/fullroutes", map_size_limit=10000) as db:
    for i, sample in enumerate(fullroutes):
        file = sample["file"]

        print(file)
        if file.endswith(".brml"):
            data = read_brml(file)
        elif file.endswith(".txt") or file.endswith(".csv"):
            data = read_txt(file)
        data = process_xrd_data(data)

        print(data.head())

        label = np.array([ROUTES.index(sample["label"])])
        data = data.to_numpy()[..., np.newaxis].transpose(2, 0, 1)

        print(data.shape, label.shape)

        db.put_samples("label", label, "data", data)

# make startingmat lmdb
with px.Writer(dirpath="./data/startingmat", map_size_limit=10000) as db:
    for i, sample in enumerate(startingmat):
        file = sample["file"]

        print(file)
        if file.endswith(".brml"):
            data = read_brml(file)
        elif file.endswith(".txt") or file.endswith(".csv"):
            data = read_txt(file)
        data = process_xrd_data(data)

        print(data.head())

        label = np.array([STARTMAT.index(sample["label"])])
        data = data.to_numpy()[..., np.newaxis].transpose(2, 0, 1)

        db.put_samples("label", label, "data", data)

routes_dataset = pxt.TorchDataset("data/fullroutes")
startingmat_dataset = pxt.TorchDataset("data/startingmat")

print(routes_dataset)
print(startingmat_dataset)
