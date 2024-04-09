import glob
import json

import numpy as np
import pyxis as px
import pyxis.torch as pxt

from utils import TransformTorchDataset, process_xrd_data, read_brml, read_txt

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


def write_samples(list_of_samples, db, label_map):
    for sample in list_of_samples:
        file = sample["file"]

        if file.endswith(".brml"):
            data = read_brml(file)
        elif file.endswith(".txt") or file.endswith(".csv"):
            data = read_txt(file)
        data = process_xrd_data(data)

        print(data.head())
        # drop "twotheta" column
        data = data.drop(columns=["TwoTheta"])

        label = np.array([label_map.index(sample["label"])])
        data = data.to_numpy().transpose(1, 0).astype(np.float32)

        db.put_samples({"data": data, "label": label})


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

# split into train and val sets
np.random.shuffle(fullroutes)
np.random.shuffle(startingmat)

train_fullroutes = fullroutes[: int(0.8 * len(fullroutes))]
val_fullroutes = fullroutes[int(0.8 * len(fullroutes) + 1) :]

train_startingmat = startingmat[: int(0.8 * len(startingmat))]
val_startingmat = startingmat[int(0.8 * len(startingmat) + 1) :]

# make fullroutes lmdb
with px.Writer(dirpath="./data/fullroutes/train", map_size_limit=10000) as db:
    write_samples(train_fullroutes, db, ROUTES)

with px.Writer(dirpath="./data/fullroutes/val", map_size_limit=10000) as db:
    write_samples(val_fullroutes, db, ROUTES)

# make startingmat lmdb
with px.Writer(dirpath="./data/startingmat/train", map_size_limit=10000) as db:
    write_samples(train_startingmat, db, STARTMAT)

with px.Writer(dirpath="./data/startingmat/val", map_size_limit=10000) as db:
    write_samples(val_startingmat, db, STARTMAT)

routes_dataset = TransformTorchDataset("data/fullroutes/train")
routes_dataset_val = TransformTorchDataset("data/fullroutes/val")
startingmat_dataset = TransformTorchDataset("data/startingmat/train")
startingmat_dataset_val = TransformTorchDataset("data/startingmat/val")

print(routes_dataset)
print(routes_dataset_val)
print(startingmat_dataset)
print(startingmat_dataset_val)
