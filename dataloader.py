from sklearn import model_selection
from torch.utils.data import DataLoader
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


    #train_df = balance_dataset(df, config)
    #train_df = balance_dataset(train_df, config, aug_factor=1)
    train_df.set_index("id", inplace=True)
    valid_df.set_index("id", inplace=True)
    test_df.set_index("id", inplace=True)

    save_df = True
    if save_df:
        save_location = config["save_location"]
        train_dataframe_location = os.path.join(save_location, 'train_df_seed' + str(config["seed"]) + '.xlsx')
        print(train_dataframe_location)
        train_df.to_excel(train_dataframe_location, index=True)

        valid_dataframe_location = os.path.join(save_location, 'valid_df_seed' + str(config["seed"]) + '.xlsx')
        print(valid_dataframe_location)
        valid_df.to_excel(valid_dataframe_location, index=True)

        test_dataframe_location = os.path.join(save_location, 'test_df_seed' + str(config["seed"]) + '.xlsx')
        print(test_dataframe_location)
        test_df.to_excel(test_dataframe_location, index=True)

    #print(fail)
    load_df_from_preset_location = False
    if load_df_from_preset_location:
        train_loc = os.path.join(dir_base, 'Zach_Analysis/result_logs/discrepancy_detection/second_labeling_batch/data_folder/seed' +str(config["seed"]) + '/train_df_seed' +str(config["seed"]) + '.xlsx')
        train_df = pd.read_excel(train_loc, engine='openpyxl')
        valid_loc = os.path.join(dir_base,'Zach_Analysis/result_logs/discrepancy_detection/second_labeling_batch/data_folder/seed' +str(config["seed"]) + '/valid_df_seed' +str(config["seed"]) + '.xlsx')
        valid_df = pd.read_excel(valid_loc, engine='openpyxl')
        test_loc = os.path.join(dir_base,'Zach_Analysis/result_logs/discrepancy_detection/second_labeling_batch/data_folder/seed' +str(config["seed"]) + '/test_df_seed' +str(config["seed"]) + '.xlsx')
        test_df = pd.read_excel(test_loc, engine='openpyxl')

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

    y_train_indices = training_set.indices

    y_train = [training_set.targets[i] for i in y_train_indices]

    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

    print(f"class sample count: {class_sample_count}")

    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **test_params)
    test_loader = DataLoader(test_set, **test_params)

    return training_loader, valid_loader, test_loader

