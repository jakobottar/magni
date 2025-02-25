import io
import math
import os
import warnings
import xml.etree.ElementTree as ET
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from mp_api.client import MPRester
from PIL import Image
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import IStructure, Lattice, Structure
from torch.utils.data import Dataset

COLS = "?,??,TwoTheta,Theta,Intensity"
## interpolation constants
X_MIN = 10
X_MAX = 70
NUM_POINTS = 4096

# cloudy-lemon-birman

# for synthetic data
FWHM = 0.2
SIGMA = FWHM * 0.42463  # 1/(2*sqrt(2*ln2))=0.42463
OMEGA = 0.001
SHIFT_SIGMA = 0.05

ROUTES = [
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

FINALMATS = ["U3O8", "UO2", "UO3"]


class RandomNoiseTransform:
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, data):
        noise = torch.randn_like(data) * self.noise_level
        return data + noise


class PeakHeightShiftTransform:
    def __init__(self, shift_scale=0.1):
        self.shift_scale = shift_scale
        self.prominence = 0.025
        self.norm_len = 15
        self.norm_scale = 0.5

        rv = scipy.stats.norm(scale=self.norm_scale)
        x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), self.norm_len)
        self.shift = rv.pdf(x)

    def __call__(self, data):

        # squeeze data
        data = data.squeeze()

        # find peaks
        index_data, _ = scipy.signal.find_peaks(np.array(data), prominence=self.prominence)
        for loc in index_data:
            # add shift to signal at index data location
            # take the chunk and multiply by shift value so we "scale" shift by the peak height
            chunk = data[loc - math.floor(self.norm_len / 2) : loc + math.ceil(self.norm_len / 2)]

            # handle out of bounds cases
            if loc + math.ceil(self.norm_len / 2) > len(data):
                shift_copy = self.shift[: len(chunk)]
                chunk *= shift_copy * np.random.normal(0, self.shift_scale)
            elif loc - math.floor(self.norm_len / 2) < 0:
                shift_copy = self.shift[-len(chunk) :]
                chunk *= shift_copy * np.random.normal(0, self.shift_scale)
            else:
                chunk *= self.shift * np.random.normal(0, self.shift_scale)

            data[loc - math.floor(self.norm_len / 2) : loc + math.ceil(self.norm_len / 2)] += chunk

        # unsqueeze data to original form
        data = data[np.newaxis, :]

        return data


class Normalize:
    def __init__(self):
        self.xpts = np.linspace(X_MIN, X_MAX, NUM_POINTS)

    def __call__(self, data):
        auc = np.trapz(data, self.xpts).astype(np.float32)
        return data / auc


def XRDtoPDF(xrd, min_angle, max_angle):

    thetas = np.linspace(min_angle / 2.0, max_angle / 2.0, len(xrd))
    Q = np.array([4 * math.pi * math.sin(math.radians(theta)) / 1.5406 for theta in thetas])
    S = np.array(xrd).flatten()

    pdf = []
    R = np.linspace(1, 40, 1000)  # Only 1000 used to reduce compute time
    integrand = Q * S * np.sin(Q * R[:, np.newaxis])

    pdf = 2 * np.trapz(integrand, Q) / math.pi

    return R, pdf


def parse_xml_file(filename) -> pd.DataFrame:

    ## load XML file
    root = ET.parse(filename).getroot()

    data = [COLS]
    ## the data we want is located at the following path in the XML file:
    for datapoint in root.findall("./DataRoutes/DataRoute/Datum"):
        data.append(datapoint.text)

    ## group into dataframe (the data is already in CSV format, so we can just read it as CSV)
    data = pd.read_csv(io.StringIO("\n".join(data)))

    return data


def read_raw(filename) -> pd.DataFrame:

    print(
        "Not feasable, the file format is not publically available\n use 'jade pattern converter' instead: https://www.icdd.com/jade-pattern-converter/"
    )

    return None


def read_brml(filename, tmp_folder="./tmp") -> pd.DataFrame:

    ## make tmp folder
    os.makedirs(tmp_folder, exist_ok=True)

    ## unzip the file into tmp folder
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(tmp_folder)

    ## read the RawData0.xml file
    data = parse_xml_file(os.path.join(tmp_folder, "Experiment0", "RawData0.xml"))

    # drop dummy variable columns
    data = data.drop(columns=["?", "??"])

    ## clean up tmp folder
    os.system(f"rm -rf {tmp_folder}")

    ## return the data
    return data


def read_txt(filename) -> pd.DataFrame:

    ## read the first line of the text file:
    with open(filename, "r") as f:
        first_line = f.readline()

        ## there's a couple different txt file formats in here
        if ";RAW4.00" in first_line:
            data_start_line = [i for i, l in enumerate(f) if l.rstrip() == "[Data]"][0] + 3  # skip 3 lines past [Data]
            data = pd.read_csv(
                filename,
                skiprows=data_start_line,
                sep=",",
                names=["TwoTheta", "Intensity", "x"],
            )
            data = data.drop(columns=["x"])

        elif "[Measurement conditions]" in first_line:
            data_start_line = [i for i, l in enumerate(f) if l.rstrip() == "[Scan points]"][
                0
            ] + 3  # skip 3 lines past [Scan points]
            data = pd.read_csv(
                filename,
                skiprows=data_start_line,
                sep=",",
                names=["TwoTheta", "Intensity", "x"],
            )
            data = data.drop(columns=["x"])
            # print(data.head())

        elif "[°2Th.]" in first_line:
            data = pd.read_csv(
                filename,
                sep="\t",
                skiprows=1,
                names=["TwoTheta", "Intensity", "x", "xx", "xxx", "xxxx", "xxxxx"],
            )
            data = data.drop(columns=["x", "xx", "xxx", "xxxx", "xxxxx"])

        elif "Angle" in first_line or "Commander Sample ID" in first_line:
            data = pd.read_csv(filename, sep=r"\s+", skiprows=1, names=["TwoTheta", "Intensity"])

        elif "PDF#" in first_line:
            data = pd.read_csv(filename, sep=r"\s+", skiprows=16)
            data = data.loc[:, ["2-Theta", "I(f)"]].rename(columns={"2-Theta": "TwoTheta", "I(f)": "Intensity"})

    return data


class SyntheticXRDGenerator:
    def __init__(self, root: str = "data", source: str = "cif", peak_pos_shift: bool = False) -> None:

        self.root = root
        self.source = source
        self.peak_pos_shift = peak_pos_shift
        self.pattern_cache = {}

    def process_xrd_data(self, data: pd.DataFrame) -> pd.DataFrame:

        ## subtract the minimum intensity from all intensities
        data["Intensity"] = data["Intensity"] - data["Intensity"].min()

        ## drop datapoints greather than 70 degrees or less than 10 degrees
        data = data[(data["TwoTheta"] >= 10) & (data["TwoTheta"] <= 70)]

        ## interpolate the data to match a standard range of 10 to 90 degrees
        x = np.linspace(X_MIN, X_MAX, NUM_POINTS)  # delta of 0.1953125 degrees
        y = np.interp(x, data["TwoTheta"], data["Intensity"])

        # copy data to new dataframe
        data = pd.DataFrame({"TwoTheta": x, "Intensity": y})

        ## normalize the intensity by integrating the area under the curve
        # data["Intensity"] = data["Intensity"] / data["Intensity"].sum()

        return data

    def df_2_np(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(df["Intensity"]).astype(np.float32)

    def generate_synthetic_xrd(self, material: str) -> pd.DataFrame:

        if material in self.pattern_cache.keys():
            intensity = self.pattern_cache[material]
            x = np.linspace(X_MIN, X_MAX, num=NUM_POINTS, endpoint=True)

        else:
            if self.source == "cif":
                filename = os.path.join(self.root, f"{material}.cif")
                structure = Structure.from_file(filename, primitive=False, sort=False, merge_tol=0.0)

            elif self.source == "mp":
                with MPRester(api_key=os.getenv("MP_API_KEY")) as mpr:
                    structure = mpr.get_structure_by_material_id(material)

            xrd = XRDCalculator(
                wavelength="CuKa", symprec=1.0
            )  # initiate XRD calculator (can specify various options here)

            pattern = xrd.get_pattern(structure, scaled=False, two_theta_range=(X_MIN, X_MAX))

            if self.peak_pos_shift:
                # shift the peak positions using a gaussian distribution
                pattern.x += np.random.normal(0, SHIFT_SIGMA, len(pattern.x))

            # spread out the peaks using a gaussian distribution
            # https://github.com/Ying-Ying-Zhang/xrd_plot/blob/main/xrd_plot.py

            a = 1 / (SIGMA * np.sqrt(2 * np.pi))
            x = np.linspace(X_MIN, X_MAX, num=NUM_POINTS, endpoint=True)

            def spectrum(x, y, x_range):
                gE = []
                for xi in x_range:
                    tot = 0
                    for xj, o in zip(x, y):
                        # ignore peaks too far out of range
                        if abs(xj - xi) > 10:
                            continue

                        L = (FWHM / (2 * np.pi)) * (1 / ((xj - xi) ** 2 + 0.25 * FWHM**2))
                        G = a * np.exp(-((xj - xi) ** 2) / (2 * SIGMA**2))
                        P = OMEGA * G + (1 - OMEGA) * L
                        tot += o * P
                    gE.append(tot)
                return gE

            intensity = spectrum(pattern.x, pattern.y, x)

            self.pattern_cache[material] = intensity

        # convert to standard pd dataframe
        df = pd.DataFrame(
            {
                "TwoTheta": x,
                "Intensity": intensity,
            }
        )

        return self.df_2_np(self.process_xrd_data(df))


class PairedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        fold_num: int = 1,
        sem_transform=None,
        xrd_transform=None,
        synthetic_xrd: bool = False,
        mode: str = "paired",
    ):
        self.root = root
        self.split = split
        self.sem_transform = sem_transform
        self.xrd_transform = xrd_transform
        self.synthetic_xrd = synthetic_xrd
        self.synxrd_generator = SyntheticXRDGenerator(root=root, peak_pos_shift=False)
        self.mode = mode  # can be 'paired', 'sem', 'xrd'

        # load dataset metadata file
        try:
            self.df = pd.read_csv(os.path.join(self.root, self.split, "metadata.csv"))
            self.df = pd.read_csv(os.path.join(self.root, self.split, "metadata.csv"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Dataset {self.root} does not exist, make sure it has been built.") from exc

        # filter out uo3 materials
        self.df = self.df[self.df["finalmat"] != "UO3"]

        # filter metadata by fold number
        # if split == "train":
        #     self.df = self.df[self.df["fold"] != fold_num]
        # else:
        #     self.df = self.df[self.df["fold"] == fold_num]

        if self.mode == "paired":
            # convert route to label
            self.df["label"] = self.df["route"].apply(ROUTES.index)
        elif self.mode == "sem":
            # convert route to label
            self.df["label"] = self.df["route"].apply(ROUTES.index)
        elif self.mode == "xrd":
            # convert route to label
            self.df["label"] = self.df["finalmat"].apply(FINALMATS.index)
            # drop duplicates of xrd_file column
            self.df = self.df.drop_duplicates(subset=["xrd_file"])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.classes = FINALMATS if self.mode == "xrd" else ROUTES
        self.str_classes = [self.classes[i] for i in self.df["label"].unique()]
        self.num_classes = len(self.df["label"].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]

        # get sample
        if self.mode == "paired" or self.mode == "sem":
            sem = Image.open(os.path.join(self.root, self.split, sample["sem_file"])).convert("RGB")
            sem = Image.open(os.path.join(self.root, self.split, sample["sem_file"])).convert("RGB")
            if self.sem_transform:
                sem = self.sem_transform(sem)

        if self.mode == "paired" or self.mode == "xrd":
            # get xrd data
            if self.synthetic_xrd:
                xrd = self.synxrd_generator.generate_synthetic_xrd(material=sample["finalmat"])
            else:
                xrd = np.load(os.path.join(self.root, self.split, sample["xrd_file"]))

            # apply xrd transform
            if self.xrd_transform:
                xrd = self.xrd_transform(xrd)

        # get label
        label = np.int64(sample["label"])

        # return data
        if self.mode == "xrd":
            return xrd, label
        elif self.mode == "sem":
            return sem, label
        else:  # paired mode
            return xrd, sem, label

    def __repr__(self):
        return f"PairedDataset: {self.split} split with {self.__len__()} samples"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy
    from torchvision.transforms import v2

    xrd_transform = v2.Compose(
        [
            torch.from_numpy,
            Normalize(),
            # PeakHeightShiftTransform(shift_scale=0.15),
            # RandomNoiseTransform(noise_level=0.005),
        ]
    )

    xrd_transform2 = v2.Compose(
        [
            torch.from_numpy,
            Normalize(),
            PeakHeightShiftTransform(shift_scale=0.2),
            RandomNoiseTransform(noise_level=0.002),
        ]
    )

    syn_dataset = PairedDataset(
        root="/usr/sci/scratch_nvme/jakobj/nfs/paired-xrd-sem-2",
        split="val",
        xrd_transform=xrd_transform,
        synthetic_xrd=True,
        mode="xrd",
    )
    syn_transf_dataset = PairedDataset(
        root="/usr/sci/scratch_nvme/jakobj/nfs/paired-xrd-sem-2",
        split="val",
        xrd_transform=xrd_transform2,
        synthetic_xrd=True,
        mode="xrd",
    )
    real_dataset = PairedDataset(
        root="/usr/sci/scratch_nvme/jakobj/nfs/paired-xrd-sem-2",
        split="val",
        xrd_transform=xrd_transform,
        synthetic_xrd=False,
        mode="xrd",
    )

    for i in range(len(syn_dataset)):
        SAMPLE_IDX = i

        syn_sample = syn_dataset[SAMPLE_IDX][0].numpy()
        syn_transf_sample = syn_transf_dataset[SAMPLE_IDX][0].numpy()
        real_sample = real_dataset[SAMPLE_IDX][0].numpy()

        print(f"Synthetic XRD - {syn_dataset.df.iloc[SAMPLE_IDX]['xrd_file']}")

        # make a 1x3 column of plots
        fig, axs = plt.subplots(3, 1, figsize=(5, 6))

        fig.suptitle(f"XRD Peaks: Synthetic,Synthetic w/Transforms, and Real")
        axs[0].plot(np.linspace(X_MIN, X_MAX, NUM_POINTS), syn_sample.squeeze())
        # axs[0].set_title("Synthetic XRD")
        axs[0].set_ylabel("Intensity")

        axs[1].plot(np.linspace(X_MIN, X_MAX, NUM_POINTS), syn_transf_sample.squeeze())
        # axs[1].set_title("Synthetic XRD w/Transforms")
        axs[1].set_ylabel("Intensity")

        axs[2].plot(np.linspace(X_MIN, X_MAX, NUM_POINTS), real_sample.squeeze())
        axs[2].set_xlabel("2θ")
        axs[2].set_ylabel("Intensity")
        # axs[2].set_title("Real XRD")

        plt.savefig("peaks.png")

        # plt.plot(np.linspace(X_MIN, X_MAX, NUM_POINTS), syn_sample.squeeze())
        # plt.title(f"Synthetic XRD - {syn_dataset.df.iloc[SAMPLE_IDX]['route']}")
        # fg

        # plt.plot(np.linspace(X_MIN, X_MAX, NUM_POINTS), syn_transf_sample.squeeze())
        # plt.title(f"Synthetic XRD w/Transforms - {syn_transf_sample.df.iloc[SAMPLE_IDX]['route']}")
        # plt.savefig("peaks_syn_transf.png")
        # plt.clf()

        # plt.plot(np.linspace(X_MIN, X_MAX, NUM_POINTS), real_sample.squeeze())
        # plt.title(f"Real XRD - {real_dataset.df.iloc[SAMPLE_IDX]['route']}")
        # plt.savefig("peaks_real.png")
        # plt.clf()

        # wait on key press
        input("Press Enter to continue...")
        plt.close()
