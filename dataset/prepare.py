from utils.config import config
import os
import pandas as pd
import csv

if __name__ == "__main__":
    dataset = pd.read_csv(os.path.join(config.dataset.root, config.dataset.train), sep="\t")
    unsupervised_dataset = dataset.sample(frac=1-(config.dataset.percent_split/100.0), random_state=100)
    supervised_dataset = dataset[~dataset.index.isin(unsupervised_dataset.index)]
    unsupervised_path = os.path.join(config.dataset.root, config.dataset.unsupervised_train)
    supervised_path = os.path.join(config.dataset.root, config.dataset.supervised_train)
    unsupervised_dataset.to_csv(unsupervised_path, sep="\t", quoting=csv.QUOTE_NONE, header=True, index=False)
    supervised_dataset.to_csv(supervised_path, sep="\t", quoting=csv.QUOTE_NONE, header=True, index=False)
    print(f"Unsupervised set created at {unsupervised_path} with {len(unsupervised_dataset)} files.")
    print(f"Supervised set created at {supervised_path} with {len(supervised_dataset)} files.")