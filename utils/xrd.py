import io
import os
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

COLS = "?,??,TwoTheta,Theta,Intensity"
## interpolation constants
X_MIN = 10
X_MAX = 90
NUM_POINTS = 4096


def parse_xml_file(filename) -> pd.DataFrame:

    ## load XML file
    root = ET.parse(filename).getroot()

    data = [COLS]
    ## the data we want is located at the following path in the XML file:
    for datapoint in root.findall("./DataRoutes/DataRoute/Datum"):
        data.append(datapoint.text)

    ## group into dataframe (the data is already in CSV format, so we can just read it as CSV)
    data = pd.read_csv(io.StringIO('\n'.join(data)))

    return data

def read_brml(filename, tmp_folder="./tmp") -> pd.DataFrame:

    ## make tmp folder
    os.makedirs(tmp_folder, exist_ok=True)

    ## unzip the file into tmp folder
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(tmp_folder)

    ## read the RawData0.xml file
    data = parse_xml_file(os.path.join(tmp_folder, "Experiment0", "RawData0.xml"))

    # drop dummy variable columns
    data = data.drop(columns=["?", "??"])

    ## clean up tmp folder
    os.system(f"rm -rf {tmp_folder}")

    ## return the data
    return data

def process_xrd_data(data: pd.DataFrame) -> pd.DataFrame:

    ## subtract the minimum intensity from all intensities
    data['Intensity'] = data['Intensity'] - data['Intensity'].min()

    ## drop datapoints greather than 90 degrees or less than 10 degrees
    data = data[(data['TwoTheta'] >= 10) & (data['TwoTheta'] <= 90)]

    ## interpolate the data to match a standard range of 10 to 90 degrees
    x = np.linspace(X_MIN, X_MAX, NUM_POINTS) # delta of 0.1953125 degrees
    y = np.interp(x, data['TwoTheta'], data['Intensity'])

    # copy data to new dataframe
    data = pd.DataFrame({"TwoTheta": x, "Intensity": y})

    ## normalize the intensity by integrating the area under the curve
    data['Intensity'] = data['Intensity'] / data['Intensity'].sum()

    return data
