# downloading the MPC dataset
import os
import requests

def download_mpc():
    ## the dataset should be downloaded in the current directory (fewshots_mpc/)
    url = 'https://github.com/sashank06/MPC-Corpus/raw/master/MPC%20CORPUS.zip'
    filename = 'mpc.zip'
    localfile = os.path.join("fewshots_mpc/", filename)
    if not os.path.exists(localfile):
        print('Downloading MPC dataset...')
        response = requests.get(url, stream=True)
        with open(localfile, 'wb') as f:
            f.write(response.content)

# unzip it
import zipfile

def unzip_mpc():
    filename = 'mpc.zip'
    localfile = os.path.join("fewshots_mpc/", filename)
    with zipfile.ZipFile(localfile, 'r') as zip_ref:
        zip_ref.extractall("fewshots_mpc/")

if __name__ == '__main__':
    download_mpc()
    print('Download complete.')
    unzip_mpc()
    print('Unzip complete.')