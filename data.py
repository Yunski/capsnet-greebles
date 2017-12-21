import argparse
import os
import sys
import gzip
import shutil
import urllib.request
import zipfile
import ruamel.yaml as yaml

from config import cfg

def download_and_extract(save_dir, data_url, update=False):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(save_dir, filename)
    splitext = os.path.splitext(filepath)
    dest, ext = splitext[0], splitext[1]

    def download_progress(count, block_size, total_size):
        sys.stdout.write("\r>> {} - Download progress: {:.1f}%".format(filename, count * block_size / total_size * 100))
        sys.stdout.flush()

    if not update and os.path.exists(filepath):
        print("{} already exists.".format(filename))
    else:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, download_progress)
        print()
        print("Successfully downloaded {}.".format(filename))
    
    if ext == '.gz':
        with gzip.open(filepath, 'rb') as f_in, open(dest, 'wb') as f_out:
            print("Extracting {}".format(filename))
            shutil.copyfileobj(f_in, f_out)
            print("Finished extracting.")
    elif ext == '.zip':
        print("Extracting {}".format(filename))
        zip_file = zipfile.ZipFile(filepath)
        zip_file.extractall(os.path.dirname(dest))
        print("Finished extracting.")
    else:
        raise ValueError("Unsupported file")        

def get_data(save_dir, urls, update=False):
    for url in urls:
        download_and_extract(save_dir, url, update)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare datasets')
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--update", default=False, type=bool) 
    args = parser.parse_args()    
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    save_dir = os.path.join(args.data_dir, cfg.dataset)
    with open(cfg.dataset_file) as stream:
        try:
            datasets = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            raise Exception(exc)
    dataset = cfg.dataset        
    if dataset not in datasets:
        raise ValueError("{} not in dataset file".format(dataset))
    selected_dataset = datasets[dataset]
    urls = [os.path.join(selected_dataset['base'], filepath) for filepath in selected_dataset['files']]
    get_data(save_dir, urls, update=args.update)

