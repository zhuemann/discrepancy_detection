import argparse
import os
import numpy as np

import pandas as pd
from train_discrepancy_detection import train_discrepancy_detection
#from train_discrepancy_detection_random_test_data import train_discrepancy_detection

from next_setence_prediction import train_discrepancy_detection_nsp
from inference_on_all_data import inference_on_all_data
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from discrepancy_datasetup import discrepancy_datasetup
from data_cleaning import count_duplicates, pick_test_set

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

    config = {"seed": 1, "batch_size": 16, "dir_base": directory_base, "epochs": 15, "n_classes": 2, "LR": 5e-6,
                  "train_samples": .75, "valid_samples": .4, "data_path": "D:/candid_ptx/"} #was .8 .5 lr was 1e-5 5e-6 5e-6 is best

    # best results for far are with lr 5e-6 and 20 epochs
    #seeds = [98, 117, 295, 456, 712, 915, 1367]
    #for seed in seeds:

    #    config["seed"] = seed
    #    folder_name = "seed" + str(config["seed"]) + "/"
    #    save_string = "/UserData/Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/radbert_final_model_first_train_pretrain_v83/" + folder_name
    #    save_location = os.path.join(directory_base, save_string)
    #    config["save_location"] = save_location
    #    inference_on_all_data(config)
    #print(fail)

    #discrepancy_datasetup(config)
    #count_duplicates(config)
    #pick_test_set(config)
    #print(fail)
    seeds = [98, 117, 295, 456, 712, 915, 1367]
    #seeds = [295, 98, 456, 915, 1367, 712]
    #seeds = [712]

    #seeds = [456, 712, 915, 1367]
    #seeds = [915, 1367, 712]
    #seeds = [117]
    #seeds = [98, 117, 295]
    #seeds = [712]
    accur_list = []

    for seed in seeds:

        folder_name = "seed" + str(seed) + "/"
        save_string = "/UserData/Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/radbert_trained_on_first_second_set_pretrain_v84/" + folder_name
        save_location = os.path.join(directory_base, save_string)

        config["seed"] = seed
        config["save_location"] = save_location

        #acc, valid_log = train_discrepancy_detection_nsp(config)
        acc, valid_log = train_discrepancy_detection(config)

        df = pd.DataFrame(valid_log)
        df["test_accuracy"] = acc
        accur_list.append(acc)

        filepath = os.path.join(config["save_location"], "validation_scores_seed" + str(seed) + '.xlsx')
        df.to_excel(filepath, index=False)

        #inference_on_all_data(config)

    print(np.mean(accur_list))