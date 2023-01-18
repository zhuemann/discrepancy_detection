from sklearn import model_selection
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import torch

from discrepancy_datasetup import balance_dataset
class TextDataset(Dataset):

    def __init__(self, dataframe, tokenizer, dir_base):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text1 = dataframe.impression1
        self.text2 = dataframe.impression2
        self.targets = self.data.label
        self.row_ids = self.data.index
        self.max_len = 512

        #self.df_data = dataframe.values
        self.data_path = os.path.join(dir_base, "public_datasets/candid_ptx/dataset1/dataset/")
        self.dir_base = dir_base

    def __len__(self):
        return len(self.text1)


    def __getitem__(self, index):
        # text extraction
        #global img, image

        text1 = str(self.text1[index])
        text1 += str(self.text2[index])
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


def setup_dataloader(df, config, tokenizer):

    seed = config["seed"]
    dir_base = config["dir_base"]
    BATCH_SIZE = config["batch_size"]
    # Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        df, train_size=config["train_samples"], random_state=seed, shuffle=True  # stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=config["test_samples"], random_state=seed, shuffle=True
        # stratify=test_valid_df.label.values
    )

    #train_df = balance_dataset(df, config)
    #train_df = balance_dataset(train_df, config, aug_factor=2)
    train_df.set_index("id", inplace=True)
    valid_df.set_index("id", inplace=True)
    test_df.set_index("id", inplace=True)

    training_set = TextDataset(train_df, tokenizer, dir_base=dir_base)
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

    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **test_params)
    test_loader = DataLoader(test_set, **test_params)

    return training_loader, valid_loader, test_loader
