import argparse
import os
import numpy as np

import pandas as pd
from train_discrepancy_detection import train_discrepancy_detection
from next_setence_prediction import train_discrepancy_detection_nsp
from inference_on_all_data import inference_on_all_data
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from discrepancy_datasetup import discrepancy_datasetup

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

    config = {"seed": 1, "batch_size": 4, "dir_base": directory_base, "epochs": 20, "n_classes": 2, "LR": 5e-6,
                  "train_samples": .75, "valid_samples": .4, "data_path": "D:/candid_ptx/"} #was .8 .5 lr was 1e-5 5e-6 5e-6 is best

    #config["seed"] = 456
    #inference_on_all_data(config)
    #print(fail)
    #discrepancy_datasetup(config)
    seeds = [117, 295, 98, 456, 915, 1367, 712]
    #seeds = [117, 295]
    #seeds = [915, 1367, 712]
    accur_list = []

    for seed in seeds:

        folder_name = "seed" + str(seed) + "/"
        save_string = "/UserData/Zach_Analysis/result_logs/discrepancy_detection/bert_cls_v33/" + folder_name
        save_location = os.path.join(directory_base, save_string)

        config["seed"] = seed
        config["save_location"] = save_location

        acc, valid_log = train_discrepancy_detection_nsp(config)
        #acc, valid_log = train_discrepancy_detection(config)

        df = pd.DataFrame(valid_log)
        df["test_accuracy"] = acc
        accur_list.append(acc)

        filepath = os.path.join(config["save_location"], "valid_150ep_seed" + str(seed) + '.xlsx')
        df.to_excel(filepath, index=False)

        #inference_on_all_data(config)

    print(np.mean(accur_list))