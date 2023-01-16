import os
import pandas as pd
from transformers import T5Model, T5Tokenizer, RobertaModel, RobertaTokenizer, AutoTokenizer
import nltk
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from discrepancy_datasetup import discrepancy_datasetup
from dataloader import setup_dataloader
from t5_classifier import T5Classifier
from roberta_classifier import RobertaClassifier
from discrepancy_datasetup import balance_dataset

def train_discrepancy_detection(config):
    nltk.download('punkt')
    dir_base = config["dir_base"]
    need_setup = False
    if need_setup:
        df = discrepancy_datasetup(config)
        save_path = os.path.join(dir_base, 'Zach_Analysis/discrepancy_data/second_set_binary_discrepancy_balanced_df.xlsx')
        df.to_excel(save_path, index=False)

    #print(df)
    #dir_base = config["dir_base"]
    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/discrepancy_data/second_set_binary_discrepancy_df.xlsx')
    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/discrepancy_data/second_set_binary_discrepancy_balanced_df.xlsx')

    df = pd.read_excel(dataframe_location, engine='openpyxl')
    #df.set_index("id", inplace=True)

    #balanced_df = balance_dataset(df)
    #print(balanced_df)
    #print(df)
    #t5_path = os.path.join(dir_base, 'Zach_Analysis/models/t5_large/')
    #tokenizer = T5Tokenizer.from_pretrained(t5_path)
    #language_model = T5Model.from_pretrained(t5_path)
    t5_path = os.path.join(dir_base, 'Zach_Analysis/roberta/')
    tokenizer = AutoTokenizer.from_pretrained(t5_path)
    language_model1 = RobertaModel.from_pretrained(t5_path)
    language_model2 = RobertaModel.from_pretrained(t5_path)
    #bert_path = ""
    #model = BertForNextSentencePrediction.from_pretrained(bert_path)


    training_loader, valid_loader, test_loader = setup_dataloader(df, config, tokenizer)
    print("after all is loaded")

    for param in language_model1.parameters():
        param.requires_grad = False

    for param in language_model2.parameters():
        param.requires_grad = False
    #model = T5Classifier(language_model, n_class=1)
    model = RobertaClassifier(language_model1, language_model2, n_class=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    LR = config["LR"]
    N_EPOCHS = config["epochs"]
    # defines which optimizer is being used
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)

    print("about to start training loop")
    lowest_loss = 100
    best_acc = 0
    valid_log = []
    avg_loss_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # vision_model.train()
        # language_model.train()
        # model_obj.train()
        model.train()
        training_accuracy = []
        #gc.collect()
        torch.cuda.empty_cache()

        loss_list = []

        for _, data in tqdm(enumerate(training_loader, 0)):

            ids1 = data['ids1'].to(device, dtype=torch.long)
            mask1 = data['mask1'].to(device, dtype=torch.long)
            token_type_ids1 = data['token_type_ids1'].to(device, dtype=torch.long)

            ids2 = data['ids2'].to(device, dtype=torch.long)
            mask2 = data['mask2'].to(device, dtype=torch.long)
            token_type_ids2 = data['token_type_ids2'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            #targets = data['targets'].to(device, dtype=torch.long)
            #targets = nn.functional.one_hot(targets)

            #targets = nn.functional.one_hot(targets)

            outputs = model(ids1, mask1, ids2, mask2, token_type_ids1, token_type_ids2)
            #outputs = model(ids1, mask1, ids2, mask2)
            # outputs = test_obj(images)
            # outputs = model_obj(images)
            outputs = torch.squeeze(outputs, dim=1)
            #targets = output_resize(targets)
            optimizer.zero_grad()

            #print(f"output size: {outputs.size()}")
            #print(outputs)
            #print(f"targets: {targets}")
            loss = criterion(outputs, targets)

            if _ % 400 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)
            #outputs = torch.round(outputs)

            # calculates the dice coefficent for each image and adds it to the list
            for i in range(0, len(outputs)):
            #    dice = dice_coeff(outputs[i], targets[i])
            #    dice = dice.item()
                #if torch.argmax(outputs[i]) == targets[i]:
                if outputs[i] == targets[i]:
                    training_accuracy.append(1)
                else:
                    training_accuracy.append(0)
            #    training_dice.append(dice)

        avg_training_accuracy = np.average(training_accuracy)
        print(f"Epoch {str(epoch)}, Average Training Accuracy = {avg_training_accuracy}")

        # each epoch, look at validation data
        with torch.no_grad():

            model.eval()
            valid_accuracy = []

            for _, data in tqdm(enumerate(valid_loader, 0)):
                #ids = data['ids'].to(device, dtype=torch.long)
                #mask = data['mask'].to(device, dtype=torch.long)
                ids1 = data['ids1'].to(device, dtype=torch.long)
                mask1 = data['mask1'].to(device, dtype=torch.long)
                token_type_ids1 = data['token_type_ids1'].to(device, dtype=torch.long)

                ids2 = data['ids2'].to(device, dtype=torch.long)
                mask2 = data['mask2'].to(device, dtype=torch.long)
                token_type_ids2 = data['token_type_ids2'].to(device, dtype=torch.long)

                #token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                #targets = data['targets'].to(device, dtype=torch.long)

                #outputs = model(ids, mask, token_type_ids)
                #outputs = model(ids1, mask1, ids2, mask2)
                outputs = model(ids1, mask1, ids2, mask2, token_type_ids1, token_type_ids2)
                outputs = torch.squeeze(outputs, dim=1)

                # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
                sigmoid = torch.sigmoid(outputs)
                outputs = torch.round(sigmoid)
                #outputs = torch.round(outputs)
                #print(outputs)
                # calculates the accuracy and adds it to the list
                for i in range(0, len(outputs)):
                    #if torch.argmax(outputs[i]) == targets[i]:
                    if outputs[i] == targets[i]:
                        valid_accuracy.append(1)
                    else:
                        valid_accuracy.append(0)

            avg_valid_acc = np.average(valid_accuracy)
            print(f"Epoch {str(epoch)}, Average Valid Accuracy = {avg_valid_acc}")
            valid_log.append(avg_valid_acc)

            if avg_valid_acc >= best_acc:
                best_acc = avg_valid_acc

                save_path = os.path.join(config["save_location"], "best_model_seed" + str(config["seed"]))
                torch.save(model.state_dict(), save_path)


    saved_path = os.path.join(config["save_location"], "best_model_seed" + str(config["seed"]))
    model.load_state_dict(torch.load(saved_path))
    model.eval()
    with torch.no_grad():
        test_accuracy = []
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids1 = data['ids1'].to(device, dtype=torch.long)
            mask1 = data['mask1'].to(device, dtype=torch.long)
            token_type_ids1 = data['token_type_ids1'].to(device, dtype=torch.long)

            ids2 = data['ids2'].to(device, dtype=torch.long)
            mask2 = data['mask2'].to(device, dtype=torch.long)
            token_type_ids2 = data['token_type_ids2'].to(device, dtype=torch.long)
            #ids = data['ids'].to(device, dtype=torch.long)
            #mask = data['mask'].to(device, dtype=torch.long)
            #token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            #targets = data['targets'].to(device, dtype=torch.float)
            targets = data['targets'].to(device, dtype=torch.long)

            #outputs = model(ids, mask, token_type_ids)
            #outputs = model(ids1, mask1, ids2, mask2)
            outputs = model(ids1, mask1, ids2, mask2, token_type_ids1, token_type_ids2)
            outputs = torch.squeeze(outputs, dim=1)
            print(f"raw outputs: {outputs}")
            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)
            #outputs = torch.round(outputs)
            print(f"predictions: {outputs}")
            print(f"targets: {targets}")
            #print(outputs)
            # calculates the accuracy and adds it to the list
            for i in range(0, len(outputs)):
                #if torch.argmax(outputs[i]) == targets[i]:
                if outputs[i] == targets[i]:
                    test_accuracy.append(1)
                else:
                    test_accuracy.append(0)

        avg_test_acc = np.average(test_accuracy)
        print(f"final test accuary: {test_accuracy}")
        print(f"Epoch {str(epoch)}, Average Test Accuracy = {avg_test_acc}")

        return avg_test_acc, valid_log