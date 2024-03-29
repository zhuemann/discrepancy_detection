from sklearn import model_selection
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset
import os
import torch
import pandas as pd
import numpy as np

from discrepancy_datasetup import balance_dataset
from discrepancy_datasetup import synonymsReplacement, shuffledTextAugmentation
class TextDataset(Dataset):

    def __init__(self, dataframe, tokenizer, dir_base, wordDict = None):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text1 = dataframe.impression1
        self.text2 = dataframe.impression2
        self.targets = self.data.label
        self.row_ids = self.data.index
        self.max_len = 512
        self.wordDict = wordDict

        #self.df_data = dataframe.values
        self.data_path = os.path.join(dir_base, "public_datasets/candid_ptx/dataset1/dataset/")
        self.dir_base = dir_base

    def __len__(self):
        return len(self.text1)


    def __getitem__(self, index):
        # text extraction
        #global img, image
        text1 = str(self.text1[index])
        text2 = str(self.text2[index])
        #if self.wordDict != None:
        #    text1 = synonymsReplacement(self.wordDict, text1)
        #    text1 = shuffledTextAugmentation(text1)
        #    text2 = synonymsReplacement(self.wordDict, text2)
        #    text2 = shuffledTextAugmentation(text2)
        text1 += text2
        text1 = " ".join(text1.split())
        text2 = str(self.text2[index])
        text2 = " ".join(text2.split())



        #print(text)
        #text = ""

        #text = text.replace("[ALPHANUMERICID]", "")
        #text = text.replace("[date]", "")
        #text = text.replace("[DATE]", "")
        #text = text.replace("[AGE]", "")

        #text = text.replace("[ADDRESS]", "")
        #text = text.replace("[PERSONALNAME]", "")
        #text = text.replace("\n", "")

        inputs1 = self.tokenizer.encode_plus(
            text1,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            #pad_to_max_length=True,
            padding= 'max_length',   #True,  # #TOD self.max_len,
            # padding='longest',
            truncation='longest_first',
            return_token_type_ids=True
        )
        ids1 = inputs1['input_ids']
        mask1 = inputs1['attention_mask']
        token_type_ids1 = inputs1["token_type_ids"]

        inputs2 = self.tokenizer.encode_plus(
            text2,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            #pad_to_max_length=True,
            padding= 'max_length',   #True,  # #TOD self.max_len,
            # padding='longest',
            truncation='longest_first',
            return_token_type_ids=True
        )
        ids2 = inputs2['input_ids']
        mask2 = inputs2['attention_mask']
        token_type_ids2 = inputs2["token_type_ids"]

        return {
            'text1' : text1,
            'ids1': torch.tensor(ids1, dtype=torch.long),
            'mask1': torch.tensor(mask1, dtype=torch.long),
            'token_type_ids1': torch.tensor(token_type_ids1, dtype=torch.long),

            'text2' : text2,
            'ids2': torch.tensor(ids2, dtype=torch.long),
            'mask2': torch.tensor(mask2, dtype=torch.long),
            'token_type_ids2': torch.tensor(token_type_ids2, dtype=torch.long),

            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'row_ids': self.row_ids[index],
        }


def setup_dataloader(df, config, tokenizer, wordDict=None):

    seed = config["seed"]
    dir_base = config["dir_base"]
    BATCH_SIZE = config["batch_size"]
    # Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        df, train_size=config["train_samples"], random_state=seed, shuffle=True,   stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=config["valid_samples"], random_state=seed, shuffle=True,
         stratify=test_valid_df.label.values
    )

    train_df = pd.concat([train_df, test_df])

    #train_df = balance_dataset(df, config)
    #train_df = balance_dataset(train_df, config, aug_factor=1)
    train_df.set_index("id", inplace=True)
    valid_df.set_index("id", inplace=True)
    test_df.set_index("id", inplace=True)

    #print(fail)
    load_df_from_preset_location = False
    if load_df_from_preset_location:
        #train_loc = os.path.join(dir_base, 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_final_train/seed' +str(config["seed"]) + '/train_df_seed' +str(config["seed"]) + '.xlsx')
        #train_loc = os.path.join(dir_base, 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated/second_and_third_labeled_df'+ '.xlsx')
        #training set
        #train_loc = os.path.join(dir_base, 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_first_train/seed' + str(config["seed"]) + '/train_df_seed' +str(config["seed"]) + '.xlsx')
        train_loc = os.path.join(dir_base, 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_first_train_first_second_labeled/seed' + str(config["seed"]) + '/train_df_seed' +str(config["seed"]) + '.xlsx')
        train_df = pd.read_excel(train_loc, engine='openpyxl')

        #valid_loc = os.path.join(dir_base,'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_finetuning/seed' +str(config["seed"]) + '/valid_df_seed' +str(config["seed"]) + '.xlsx')
        #valid_loc = os.path.join(dir_base,'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_first_train/seed' +str(config["seed"]) + '/valid_df_seed' +str(config["seed"]) + '.xlsx')
        valid_loc = os.path.join(dir_base,'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_first_train_first_second_labeled/seed' +str(config["seed"]) + '/valid_df_seed' +str(config["seed"]) + '.xlsx')
        valid_df = pd.read_excel(valid_loc, engine='openpyxl')

        test_loc = os.path.join(dir_base,'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_v1/seed' +str(config["seed"]) + '/test_df_seed' +str(config["seed"]) + '.xlsx')
        test_df = pd.read_excel(test_loc, engine='openpyxl')

    fine_tuning = True
    if fine_tuning:

        train_loc = os.path.join(dir_base,
                                 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_finetune/seed' + str(
                                     config["seed"]) + '/train_df_seed' + str(config["seed"]) + '.xlsx')
        train_df = pd.read_excel(train_loc, engine='openpyxl')

        valid_loc = os.path.join(dir_base,
                                 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_finetune/seed' + str(
                                     config["seed"]) + '/valid_df_seed' + str(config["seed"]) + '.xlsx')
        valid_df = pd.read_excel(valid_loc, engine='openpyxl')

        test_loc = os.path.join(dir_base,
                                'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_v1/seed' + str(
                                    config["seed"]) + '/test_df_seed' + str(config["seed"]) + '.xlsx')
        test_df = pd.read_excel(test_loc, engine='openpyxl')

    save_df = True
    if save_df:
        save_location = config["save_location"]
        train_dataframe_location = os.path.join(save_location, 'train_df_seed' + str(config["seed"]) + '.xlsx')
        print(train_dataframe_location)
        train_df.to_excel(train_dataframe_location, index=True)

        valid_dataframe_location = os.path.join(save_location, 'valid_df_seed' + str(config["seed"]) + '.xlsx')
        print(valid_dataframe_location)
        valid_df.to_excel(valid_dataframe_location, index=True)

        #test_dataframe_location = os.path.join(save_location, 'test_df_seed' + str(config["seed"]) + '.xlsx')
        #print(test_dataframe_location)
        #test_df.to_excel(test_dataframe_location, index=True)

    training_set = TextDataset(train_df, tokenizer, dir_base=dir_base, wordDict= wordDict)
    valid_set = TextDataset(valid_df, tokenizer, dir_base=dir_base)
    test_set = TextDataset(test_df, tokenizer, dir_base=dir_base)

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 4
                    }

    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 4
                   }

    ## added to trying sampling from training data
    #y_train_indices = training_set.indices
    #y_train_indices = range(0,len(train_df))                        #gets a list of the index 0 to lenth of df
    #y_train = [training_set.targets[i] for i in y_train_indices]    #get a list of all of the training labels
    #print(f"y train: {y_train}")
    #print(f"y train len: {len(y_train)}")
    #class_sample_count = np.array(
    #    [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]) # counts the number of each training value
    #print(type(class_sample_count))
    #print(f"class sample count: {class_sample_count}")

    #class_sample_count = np.array([1134, 94])                       #sets the counts to the values in the orginal set
    #class_sample_count = np.array([1228, 1228])
    #class_sample_count =  np.array([94, 1134])
    #class_sample_count =  np.array([94, 1134])

    #print(f"class sample count: {class_sample_count}")
    #print(type(class_sample_count))

    #class_sample_count =  [1134, 94]
    #weight = 1. / class_sample_count                    # calculates the weight for each sample
    #weight = np.array([1134/1758, 94/1758])
    #weight = np.array([1271/1762, 105/1762])
    #weight = np.array([100, 105/1762])


    #print(f"weight values: {weight}")
    #samples_weight = np.array([weight[t] for t in y_train])         # makes an array where each index is the weight to select it
    #print(f"len of sample weights: {len(samples_weight)}")
    #samples_weight = torch.from_numpy(samples_weight)
    #print(f"samples weight: {samples_weight}")
    #sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), 1368, replacement=False) # was 1228

    #y = torch.from_numpy(np.array([0, 0, 1, 1, 0, 0, 1, 1]))
    #y = torch.from_numpy(np.array(y_train))
    #sampler = StratifiedSampler(class_vector=y, batch_size=16)

    #training_loader = DataLoader(training_set, sampler=sampler, batch_size=BATCH_SIZE, num_workers=4)
    ##
    training_loader = DataLoader(training_set, **train_params)

    valid_loader = DataLoader(valid_set, **test_params)
    test_loader = DataLoader(test_set, **test_params)

    return training_loader, valid_loader, test_loader


def setup_random_training_loader(df_negative, df_positive, base_pos, base_neg, new_pos, new_neg,  config, tokenizer, wordDict=None):
    # base dataest is 1134 negatives for 94 postives

    seed = config["seed"]
    dir_base = config["dir_base"]
    BATCH_SIZE = config["batch_size"]

    #train_df_positive = df_positive.sample(n=21)
    #train_df = pd.concat([train_df_positive, df_negative])
    #train_df = pd.concat([ train_df, base_pos])
    #df_negative = df_negative.sample(n=1134)
    #df_positive = df_positive.sample(n=94)
    #train_df = pd.concat([df_negative, base_pos])

    #added_pos = new_pos.sample(11)                          #get n samples from positves cases
    #postive_df = pd.concat([base_pos, added_pos])           #add the n samples to the already postive cases
    #negative_df = pd.concat([base_neg, new_neg])            #add the new negative samples to the negative cases
    #train_df = pd.concat([postive_df, negative_df])         #create final training set

    train_df = pd.concat([base_pos, base_neg])

    training_set = TextDataset(train_df, tokenizer, dir_base=dir_base, wordDict= wordDict)


    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 4
                    }

    training_loader = DataLoader(training_set, **train_params)

    return training_loader