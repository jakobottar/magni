import io
import os
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

COLS = "?,??,TwoTheta,Theta,Intensity"

def make_dataframe(rows: list) ->pd.DataFrame:
    dataframe = pd.read_csv(io.StringIO('\n'.join(rows)))
    return dataframe

def parse_xml_file(filename):

    ## load XML file
    tree = ET.parse(filename)
    root = tree.getroot()

    data = [COLS]
    ## the data we want is located at the following path in the XML file:
    for datapoint in root.findall("./DataRoutes/DataRoute/Datum"):
        data.append(datapoint.text)

    ## group into dataframe
    data = make_dataframe(data)

    return data

def read_brml(filename, tmp_folder="./tmp"):

    ## make tmp folder
    os.makedirs(tmp_folder, exist_ok=True)

    ## unzip the file into tmp folder
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(tmp_folder)

    ## read the RawData0.xml file
    data = parse_xml_file(os.path.join(tmp_folder, "Experiment0", "RawData0.xml"))

    ## clean up tmp folder
    os.system(f"rm -rf {tmp_folder}")

    ## return the data
    return data
