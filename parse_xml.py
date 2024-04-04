import glob

import matplotlib.pyplot as plt

from utils import read_brml

## read all files in "brml_files" directory
# for file in glob.glob("brml_files/tmp/*.brml"):
for file in ["10_70.brml", "10_90.brml"]:
    print(file)
    data = read_brml(file)

    print(data.head())

    ## make a line chart of TwoTheta vs Intensity
    plt.plot(data['TwoTheta'], data['Intensity'])
    plt.savefig("xrd.png")


