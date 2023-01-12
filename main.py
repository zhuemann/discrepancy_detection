import argparse
import os

import pandas as pd
from train_discrepancy_detection import train_discrepancy_detection
from inference_on_all_data import inference_on_all_data
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def create_parser():
    parser = argparse.ArgumentParser(description="The main file to run multimodal setup. Consists of pre-training joint representation, masked language modeling and report generation.")
    parser.add_argument('--local', '-l', type=bool, help="Should the program run locally", default=False)
    parser.add_argument('--report_gen', '-r', type=bool, help="Should we train report generation?", default=False)
    parser.add_argument('--mlm_pretraining', '-m', type=bool, help="Should we perform MLM pretraining?", default=False)
    parser.add_argument('--contrastive_training', '-p', type=bool, help="Should we perform multimodal pretraining?", default=False)
    arg = parser.parse_args()
    return arg

if __name__ == '__main__':

    args = create_parser()
    # local = args.local

    local = False
    if local:
        directory_base = "Z:/"
    else:
        directory_base = "/UserData/"

    config = {"seed": 1, "batch_size": 8, "dir_base": directory_base, "epochs": 100, "n_classes": 2, "LR": 1e-4,
                  "train_samples": .8, "test_samples": .5, "data_path": "D:/candid_ptx/"}

    #inference_on_all_data(config)

    seeds = [117, 295, 98, 456, 915, 1367, 712]
    #seeds = [712]

    for seed in seeds:

        folder_name = "seed" + str(seed) + "/"
        save_string = "/UserData/Zach_Analysis/result_logs/discrepancy_detection/second_datasetv1/" + folder_name
        save_location = os.path.join(directory_base, save_string)
        save_location = ""

        config["seed"] = seed
        config["save_location"] = save_location

        acc, valid_log = train_discrepancy_detection(config)

        df = pd.DataFrame(valid_log)
        df["test_accuracy"] = acc

        filepath = os.path.join(config["save_location"], "valid_150ep_seed" + str(seed) + '.xlsx')
        df.to_excel(filepath, index=False)