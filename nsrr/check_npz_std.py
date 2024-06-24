import os
import sys

import numpy as np

project_root = ".."
sys.path.append(project_root)

DATASETS_PATH = os.path.join(project_root, "resources", "datasets", "nsrr")


if __name__ == "__main__":

    dataset_name_list = [
        "shhs1",
    ]

    top_k = 40
    epoch_samples = int(200 * 30)
    for dataset_name in dataset_name_list:
        print("Check %s" % dataset_name)
        npz_dir = os.path.abspath(
            os.path.join(DATASETS_PATH, dataset_name, "register_and_state")
        )
        all_files = os.listdir(npz_dir)
        all_files.sort()
        all_files = [f for f in all_files if ".npz" in f]
        all_fname = np.array(all_files)
        all_files = [os.path.join(npz_dir, f) for f in all_files]
        all_files = np.array(all_files)

        all_std = []
        all_n_pages = []
        all_channels = []
        for f in all_files:
            data_dict = np.load(f)
            tmp_signal = data_dict["signal"]
            tmp_std = tmp_signal.std()
            tmp_n_pages = tmp_signal.size / epoch_samples
            all_std.append(tmp_std)
            all_n_pages.append(tmp_n_pages)
            all_channels.append(data_dict["channel"])
            print("Loaded %s" % f)
        all_std = np.array(all_std)
        all_n_pages = np.array(all_n_pages)

        print("\nReport:")
        print("Subjects %d" % len(all_std))
        print(
            "STD - min %s, mean %s, max %s"
            % (all_std.min(), all_std.mean(), all_std.max())
        )
        print(
            "Pages - min %s, mean %s, max %s"
            % (all_n_pages.min(), all_n_pages.mean(), all_n_pages.max())
        )

        print("Channels found:")
        values, counts = np.unique(all_channels, return_counts=True)
        for v, c in zip(values, counts):
            print("%s: %d" % (v, c))

        print("\nSmallest STD values: (Top %d)" % top_k)
        sorted_locs = np.argsort(all_std)
        for loc in sorted_locs[:top_k]:
            print(
                "    File %s, STD %s, Pages %s"
                % (all_fname[loc], all_std[loc], all_n_pages[loc])
            )
