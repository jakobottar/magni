import glob

import matplotlib.pyplot as plt

from utils import process_xrd_data, read_brml, read_txt

## read all files in "brml_files" directory
# for file in glob.glob("brml_files/tmp/*.brml"):
for file in ["10_70.brml", "10_90.brml"]:
    print(file)
    data = process_xrd_data(read_brml(file))

    print(data.head())

    print(data['Intensity'].min(), data['Intensity'].max())
    print(data['TwoTheta'].min(), data['TwoTheta'].max())

    # ## make a line chart of TwoTheta vs Intensity
    plt.plot(data['TwoTheta'], data['Intensity'])
    plt.show()

## read all files in "txt_files" directory
# for file in glob.glob("txt_files/utf8/*.txt"):
for file in ["10_70.txt", "10_90.txt"]:
    print(file)
    data = process_xrd_data(read_txt(file))

    print(data.head())

    print(data['Intensity'].min(), data['Intensity'].max())
    print(data['TwoTheta'].min(), data['TwoTheta'].max())

    # ## make a line chart of TwoTheta vs Intensity
    plt.plot(data['TwoTheta'], data['Intensity'])
    plt.savefig("sample.png")
