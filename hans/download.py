"""Download the three MedMNIST v2 files used here (about 131 MB) into data/.

python -m hans.download
"""
import os
import urllib.request

URL = "https://zenodo.org/records/10519652/files/{}?download=1"
FILES = ["dermamnist_64.npz", "pneumoniamnist_64.npz", "breastmnist_128.npz"]

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for f in FILES:
        out = os.path.join("data", f)
        if os.path.exists(out):
            print("have", out)
            continue
        print("downloading", f)
        urllib.request.urlretrieve(URL.format(f), out)
