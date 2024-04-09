import glob
import json

import numpy as np
import pyxis as px

from utils import TransformTorchDataset, process_xrd_data, read_brml, read_txt

ROUTE = [
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

FINALMAT = [
    "U3O8",
    "UO2",
    "UO3",
]

finalmat_map = {
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
routes = []
finalmat = []
for route in ROUTE:
    files = glob.glob(f"data/{route}/*")
    for file in files:
        routes.append({"label": route, "file": file})
        finalmat.append({"label": finalmat_map[route], "file": file})


# write json files
with open("data/routes.json", "w", encoding="utf-8") as f:
    json.dump(routes, f, indent=4)

with open("data/finalmat.json", "w", encoding="utf-8") as f:
    json.dump(finalmat, f, indent=4)

# split into train and val sets
np.random.shuffle(routes)
np.random.shuffle(finalmat)

train_routes = routes[: int(0.8 * len(routes))]
val_routes = routes[int(0.8 * len(routes) + 1) :]

train_finalmat = finalmat[: int(0.8 * len(finalmat))]
val_finalmat = finalmat[int(0.8 * len(finalmat) + 1) :]

# make routes lmdb
with px.Writer(dirpath="./data/routes/train", map_size_limit=10000) as db:
    write_samples(train_routes, db, ROUTE)

with px.Writer(dirpath="./data/routes/val", map_size_limit=10000) as db:
    write_samples(val_routes, db, ROUTE)

# make finalmat lmdb
with px.Writer(dirpath="./data/finalmat/train", map_size_limit=10000) as db:
    write_samples(train_finalmat, db, FINALMAT)

with px.Writer(dirpath="./data/finalmat/val", map_size_limit=10000) as db:
    write_samples(val_finalmat, db, FINALMAT)

routes_dataset = TransformTorchDataset("data/routes/train")
routes_dataset_val = TransformTorchDataset("data/routes/val")
finalmat_dataset = TransformTorchDataset("data/finalmat/train")
finalmat_dataset_val = TransformTorchDataset("data/finalmat/val")

print(routes_dataset)
print(routes_dataset_val)
print(finalmat_dataset)
print(finalmat_dataset_val)

counter = [0, 0, 0]
mean = 0
for data, label in finalmat_dataset:
    counter[int(label)] += 1
    mean += data.mean()

print(f"train -- U3O8: {counter[0]}, UO2: {counter[1]}, UO3: {counter[2]}")
print(f"mean: {mean / len(finalmat_dataset)}")

counter = [0, 0, 0]
for _, label in finalmat_dataset_val:
    counter[int(label)] += 1

print(f"val -- U3O8: {counter[0]}, UO2: {counter[1]}, UO3: {counter[2]}")

### make xrd datasets for demo setup
### make a seperate dataset for each final material

for mat in FINALMAT:
    mat_files = [file for file in finalmat if file["label"] == mat]

    # make finalmat lmdb
    with px.Writer(dirpath=f"./data/demo-mat/{mat}", map_size_limit=10000) as db:
        write_samples(mat_files, db, FINALMAT)
