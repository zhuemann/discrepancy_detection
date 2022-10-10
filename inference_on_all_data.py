import os
import pandas as pd
from transformers import T5Model, T5Tokenizer, AutoTokenizer
import nltk
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from dataloader import TextDataset
from torch.utils.data import DataLoader
from t5_classifier import T5Classifier

#HF_DATASETS_OFFLINE = "1"
#TRANSFORMERS_OFFLINE = "1"

def inference_on_all_data(config):
    #HF_DATASETS_OFFLINE = "1"
    #TRANSFORMERS_OFFLINE = "1"
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    dir_base = config["dir_base"]

    t5_path = os.path.join(dir_base, 'Zach_Analysis/models/t5_large/')
    tokenizer = AutoTokenizer.from_pretrained(t5_path)
    #tokenizer = T5Tokenizer.from_pretrained(t5_path)
    language_model = T5Model.from_pretrained(t5_path)

    dir_base = config["dir_base"]
    dataframe_location = os.path.join(dir_base,
                                      'Zach_Analysis/discrepancy_data/first_labeled_batch.xlsx')

    df = pd.concat(pd.read_excel(dataframe_location, sheet_name=None, engine='openpyxl'), ignore_index=True)
    # df = pd.read_excel(dataframe_location, engine='openpyxl')
    df = df.dropna(axis=0, how='all')

    label_idx = 0
    data_with_labels = pd.DataFrame(columns=['id', 'impression1', 'impression2', 'label'])
    index = -1
    num_neg = 0
    for _, row in df.iterrows():

        index += 1
        if index == 0:
            continue
        #if pd.isna(row['Discrepancy']):
        #    continue

        impression1 = df.iloc[index - 1]
        impression2 = row

        if impression1['Accession Number'] == impression2['Accession Number']:
            accession = impression2['Accession Number']
            report1 = impression1['Impression']
            # print(report1)
            report2 = impression2['Impression']
            label = impression2['Discrepancy']
            #if label == 1:
            data_with_labels.loc[label_idx] = [accession, report1, report2, label]
            label_idx += 1
            #else:
            #    if num_neg < 20000:
            #        data_with_labels.loc[label_idx] = [accession, report1, report2, label]
            #        num_neg += 1
            #        label_idx += 1

    print(df)
    print(data_with_labels)

    data_df = data_with_labels
    data_df.set_index("id", inplace=True)

    test_set = TextDataset(data_df, tokenizer, dir_base=dir_base)


    test_params = {'batch_size': config["batch_size"],
                   'shuffle': True,
                   'num_workers': 4
                   }

    test_loader = DataLoader(test_set, **test_params)

    model = T5Classifier(language_model, n_class=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, 2):
        # vision_model.train()
        # language_model.train()
        # model_obj.train()
        model.train()
        training_accuracy = []
        #gc.collect()
        torch.cuda.empty_cache()

        loss_list = []

        for _, data in tqdm(enumerate(test_loader, 0)):

            ids1 = data['ids1'].to(device, dtype=torch.long)
            mask1 = data['mask1'].to(device, dtype=torch.long)
            ids2 = data['ids2'].to(device, dtype=torch.long)
            mask2 = data['mask2'].to(device, dtype=torch.long)
            #token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            #outputs = model(ids1, mask1, ids2, mask2, token_type_ids)
            outputs = model(ids1, mask1, ids2, mask2)
            # outputs = test_obj(images)
            # outputs = model_obj(images)
            outputs = torch.squeeze(outputs, dim=1)
            #targets = output_resize(targets)

            #print(f"output size: {outputs.size()}")
            #print(outputs)



            # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)

            # calculates the dice coefficent for each image and adds it to the list
            for i in range(0, len(outputs)):
            #    dice = dice_coeff(outputs[i], targets[i])
            #    dice = dice.item()
                if outputs[i] == targets[i]:
                    training_accuracy.append(1)
                else:
                    training_accuracy.append(0)
            #    training_dice.append(dice)

        avg_training_accuracy = np.average(training_accuracy)
        print(f"Epoch {str(epoch)}, Average Training Accuracy = {avg_training_accuracy}")