import os
import pandas as pd
from transformers import T5Model, T5Tokenizer, RobertaModel, RobertaTokenizer, AutoTokenizer
import nltk
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from dataloader import TextDataset
from discrepancy_datasetup import discrepancy_datasetup_second_set
from dataloader import setup_dataloader
from t5_classifier import T5Classifier
from roberta_classifier_dot_product import RobertaClassifier
from single_model_classifier import RobertaSingleClassifier
from discrepancy_datasetup import balance_dataset

def train_discrepancy_detection(config):
    nltk.download('punkt')
    dir_base = config["dir_base"]
    need_setup = False
    if need_setup:
        df = discrepancy_datasetup_second_set(config)
        #df = discrepancy_datasetup(config)
        save_path = os.path.join(dir_base, 'Zach_Analysis/discrepancy_data/used_to_train_third_model/first_training_set_relabeled.xlsx')
        df.to_excel(save_path, index=False)
        print(fail)

    #t5_path = os.path.join(dir_base, 'Zach_Analysis/models/t5_large/')
    #tokenizer = T5Tokenizer.from_pretrained(t5_path)
    #language_model1 = T5Model.from_pretrained(t5_path)
    #t5_path = os.path.join(dir_base, 'Zach_Analysis/roberta/')
    #t5_path = os.path.join(dir_base, 'Zach_Analysis/roberta_large/')
    t5_path = os.path.join(dir_base, 'Zach_Analysis/models/rad_bert/')
    tokenizer = AutoTokenizer.from_pretrained(t5_path)
    language_model1 = RobertaModel.from_pretrained(t5_path)
    #language_model1 = BertModel.from_pretrained(t5_path)
    #print("after model")
    #language_model2 = RobertaModel.from_pretrained(t5_path)

    # synonym replacement setup
    wordReplacementPath = os.path.join(config["dir_base"], 'Zach_Analysis/discrepancy_data/full_synonym_list.xlsx')

    dfWord = pd.read_excel(wordReplacementPath, engine='openpyxl')
    dfWord.set_index("word", inplace=True)

    wordDict = dfWord.to_dict()
    for key in list(wordDict["synonyms"].keys()):
        string = wordDict["synonyms"][key][2:-2]
        wordList = string.split("', '")
        wordDict["synonyms"][key] = wordList


    # setting up the first dataloader
    dir_base = config["dir_base"]
    BATCH_SIZE = config["batch_size"]

    load_df_from_preset_location = True
    if load_df_from_preset_location:
        # train_loc = os.path.join(dir_base, 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_final_train/seed' +str(config["seed"]) + '/train_df_seed' +str(config["seed"]) + '.xlsx')
        # train_loc = os.path.join(dir_base, 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated/second_and_third_labeled_df'+ '.xlsx')
        # training set
        # train_loc = os.path.join(dir_base, 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_first_train/seed' + str(config["seed"]) + '/train_df_seed' +str(config["seed"]) + '.xlsx')
        train_loc = os.path.join(dir_base,
                                 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_finetune/seed' + str(
                                     config["seed"]) + '/train_df_seed' + str(config["seed"]) + '.xlsx')
        train_loc = os.path.join(dir_base,
                                 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_first_train_first_second_labeled/seed' + str(
                                     config["seed"]) + '/train_df_seed' + str(config["seed"]) + '.xlsx')

        train_df = pd.read_excel(train_loc, engine='openpyxl')

        # valid_loc = os.path.join(dir_base,'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_finetuning/seed' +str(config["seed"]) + '/valid_df_seed' +str(config["seed"]) + '.xlsx')
        # valid_loc = os.path.join(dir_base,'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_first_train/seed' +str(config["seed"]) + '/valid_df_seed' +str(config["seed"]) + '.xlsx')
        valid_loc = os.path.join(dir_base,
                                 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_finetune/seed' + str(
                                     config["seed"]) + '/valid_df_seed' + str(config["seed"]) + '.xlsx')
        valid_loc = os.path.join(dir_base,
                                 'Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/data_folder_updated_first_train_first_second_labeled/seed' + str(
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

        # test_dataframe_location = os.path.join(save_location, 'test_df_seed' + str(config["seed"]) + '.xlsx')
        # print(test_dataframe_location)
        # test_df.to_excel(test_dataframe_location, index=True)

    training_set = TextDataset(train_df, tokenizer, dir_base=dir_base, wordDict=wordDict)
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

    print("after all is loaded")

    for param in language_model1.parameters():
        param.requires_grad = True

    #for param in language_model2.parameters():
    #    param.requires_grad = True
    #model = T5Classifier(language_model1, n_class=1)
    #model = RobertaClassifier(language_model1, language_model2, n_class=1)
    model = RobertaSingleClassifier(language_model1, n_class=1)
    for param in model.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #for param in model.parameters():
    #    param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    LR = config["LR"]
    N_EPOCHS = config["epochs"]
    # defines which optimizer is being used
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1250, eta_min=1e-7, last_epoch=-1, verbose=False) #used 2100 for run 48

    print("about to start  pretraing loop")
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
        confusion_matrix = [[0, 0], [0, 0]]

        loss_list = []
        print(scheduler.get_lr())
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

            #outputs = torch.sigmoid(outputs)
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
            scheduler.step()


            # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)
            #outputs = torch.round(outputs)

            for i in range(0, len(outputs)):
                actual = int(targets[i].detach().cpu().data.numpy())
                # predicted = outputs.argmax(dim=1)[i].detach().cpu().data.numpy()
                predicted = int(outputs[i].detach().cpu().data.numpy())
                confusion_matrix[predicted][actual] += 1

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
        print(f"Train Confusion matrix: {confusion_matrix}")

        # each epoch, look at validation data
        with torch.no_grad():

            model.eval()
            valid_accuracy = []
            confusion_matrix = [[0, 0], [0, 0]]

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
                for i in range(0, len(outputs)):
                    actual = targets[i].detach().cpu().data.numpy()
                    predicted = outputs[i].detach().cpu().data.numpy()
                    confusion_matrix[int(predicted)][int(actual)] += 1
                    if outputs[i] == targets[i]:
                        valid_accuracy.append(1)
                    else:
                        valid_accuracy.append(0)
                # calculates the accuracy and adds it to the list
                #for i in range(0, len(outputs)):
                    #if torch.argmax(outputs[i]) == targets[i]:
                #    if outputs[i] == targets[i]:
                #        valid_accuracy.append(1)
                #    else:
                #        valid_accuracy.append(0)

            avg_valid_acc = np.average(valid_accuracy)
            print(f"Epoch {str(epoch)}, Average Valid Accuracy = {avg_valid_acc}")
            valid_log.append(avg_valid_acc)
            print(f"Valid Confusion matrix: {confusion_matrix}")

            if avg_valid_acc >= best_acc:
                best_acc = avg_valid_acc
                print(f"config save location: {config['save_location']}")
                save_path = os.path.join(config["save_location"], "best_model_seed" + str(config["seed"]))
                torch.save(model.state_dict(), save_path)


# put fine tuning step here
    print(f"about to start the fine tuning step")


    fine_tune_step = True
    if fine_tune_step:
        # loading in the model we want to fine tune
        model_path_new = "/UserData/Zach_Analysis/result_logs/discrepancy_detection/third_labeling_batch/radbert_final_model_first_train_pretrain_v83/seed" + str(
            config["seed"])
        saved_path = os.path.join(model_path_new, "best_model_seed" + str(config["seed"]))
        model.load_state_dict(torch.load(saved_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # for param in model.parameters():
    #    param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    LR = 2e-6
    N_EPOCHS = 12
    # defines which optimizer is being used
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1250, eta_min=1e-7, last_epoch=-1,
                                                           verbose=False)  # used 2100 for run 48

    print("about to start fine tuning loop")
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
        # gc.collect()
        torch.cuda.empty_cache()
        confusion_matrix = [[0, 0], [0, 0]]

        loss_list = []
        print(scheduler.get_lr())
        for _, data in tqdm(enumerate(training_loader, 0)):

            ids1 = data['ids1'].to(device, dtype=torch.long)
            mask1 = data['mask1'].to(device, dtype=torch.long)
            token_type_ids1 = data['token_type_ids1'].to(device, dtype=torch.long)

            ids2 = data['ids2'].to(device, dtype=torch.long)
            mask2 = data['mask2'].to(device, dtype=torch.long)
            token_type_ids2 = data['token_type_ids2'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            # targets = data['targets'].to(device, dtype=torch.long)
            # targets = nn.functional.one_hot(targets)

            # targets = nn.functional.one_hot(targets)

            outputs = model(ids1, mask1, ids2, mask2, token_type_ids1, token_type_ids2)
            # outputs = model(ids1, mask1, ids2, mask2)
            # outputs = test_obj(images)
            # outputs = model_obj(images)
            outputs = torch.squeeze(outputs, dim=1)

            # outputs = torch.sigmoid(outputs)
            # targets = output_resize(targets)
            optimizer.zero_grad()

            # print(f"output size: {outputs.size()}")
            # print(outputs)
            # print(f"targets: {targets}")
            loss = criterion(outputs, targets)

            if _ % 400 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)
            # outputs = torch.round(outputs)

            for i in range(0, len(outputs)):
                actual = int(targets[i].detach().cpu().data.numpy())
                # predicted = outputs.argmax(dim=1)[i].detach().cpu().data.numpy()
                predicted = int(outputs[i].detach().cpu().data.numpy())
                confusion_matrix[predicted][actual] += 1

            # calculates the dice coefficent for each image and adds it to the list
            for i in range(0, len(outputs)):
                #    dice = dice_coeff(outputs[i], targets[i])
                #    dice = dice.item()
                # if torch.argmax(outputs[i]) == targets[i]:
                if outputs[i] == targets[i]:
                    training_accuracy.append(1)
                else:
                    training_accuracy.append(0)
            #    training_dice.append(dice)

        avg_training_accuracy = np.average(training_accuracy)
        print(f"Epoch {str(epoch)}, Average Training Accuracy = {avg_training_accuracy}")
        print(f"Train Confusion matrix: {confusion_matrix}")

        # each epoch, look at validation data
        with torch.no_grad():

            model.eval()
            valid_accuracy = []
            confusion_matrix = [[0, 0], [0, 0]]

            for _, data in tqdm(enumerate(valid_loader, 0)):
                # ids = data['ids'].to(device, dtype=torch.long)
                # mask = data['mask'].to(device, dtype=torch.long)
                ids1 = data['ids1'].to(device, dtype=torch.long)
                mask1 = data['mask1'].to(device, dtype=torch.long)
                token_type_ids1 = data['token_type_ids1'].to(device, dtype=torch.long)

                ids2 = data['ids2'].to(device, dtype=torch.long)
                mask2 = data['mask2'].to(device, dtype=torch.long)
                token_type_ids2 = data['token_type_ids2'].to(device, dtype=torch.long)

                # token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                # targets = data['targets'].to(device, dtype=torch.long)

                # outputs = model(ids, mask, token_type_ids)
                # outputs = model(ids1, mask1, ids2, mask2)
                outputs = model(ids1, mask1, ids2, mask2, token_type_ids1, token_type_ids2)
                outputs = torch.squeeze(outputs, dim=1)

                # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
                sigmoid = torch.sigmoid(outputs)
                outputs = torch.round(sigmoid)
                # outputs = torch.round(outputs)
                # print(outputs)
                for i in range(0, len(outputs)):
                    actual = targets[i].detach().cpu().data.numpy()
                    predicted = outputs[i].detach().cpu().data.numpy()
                    confusion_matrix[int(predicted)][int(actual)] += 1
                    if outputs[i] == targets[i]:
                        valid_accuracy.append(1)
                    else:
                        valid_accuracy.append(0)
                # calculates the accuracy and adds it to the list
                # for i in range(0, len(outputs)):
                # if torch.argmax(outputs[i]) == targets[i]:
                #    if outputs[i] == targets[i]:
                #        valid_accuracy.append(1)
                #    else:
                #        valid_accuracy.append(0)

            avg_valid_acc = np.average(valid_accuracy)
            print(f"Epoch {str(epoch)}, Average Valid Accuracy = {avg_valid_acc}")
            valid_log.append(avg_valid_acc)
            print(f"Valid Confusion matrix: {confusion_matrix}")

            if avg_valid_acc >= best_acc:
                best_acc = avg_valid_acc
                print(f"config save location: {config['save_location']}")
                save_path = os.path.join(config["save_location"], "best_model_seed" + str(config["seed"]))
                torch.save(model.state_dict(), save_path)

    saved_path = os.path.join(config["save_location"], "best_model_seed" + str(config["seed"]))
    confusion_matrix = [[0, 0], [0, 0]]
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
            # ids = data['ids'].to(device, dtype=torch.long)
            # mask = data['mask'].to(device, dtype=torch.long)
            # token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            # targets = data['targets'].to(device, dtype=torch.float)
            targets = data['targets'].to(device, dtype=torch.long)

            # outputs = model(ids, mask, token_type_ids)
            # outputs = model(ids1, mask1, ids2, mask2)
            outputs = model(ids1, mask1, ids2, mask2, token_type_ids1, token_type_ids2)
            outputs = torch.squeeze(outputs, dim=1)
            # print(f"raw outputs: {outputs}")
            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)
            # outputs = torch.round(outputs)
            # print(f"predictions: {outputs}")
            # print(f"targets    : {targets}")
            # print(outputs)

            for i in range(0, len(outputs)):
                actual = targets[i].detach().cpu().data.numpy()
                predicted = outputs[i].detach().cpu().data.numpy()
                confusion_matrix[int(predicted)][int(actual)] += 1
                if outputs[i] == targets[i]:
                    test_accuracy.append(1)
                else:
                    test_accuracy.append(0)




















        avg_test_acc = np.average(test_accuracy)
        print(f"final test accuary: {test_accuracy}")
        print(f"Epoch {str(epoch)}, Average Test Accuracy = {avg_test_acc}")
        matrix_path = os.path.join(config["save_location"], "confusion_matrix" + str(config["seed"]) + '.xlsx')
        df_matrix = pd.DataFrame(confusion_matrix)
        df_matrix.to_excel(matrix_path, index=False)
        print(f"Test Confusion matrix: {confusion_matrix}")

        return avg_test_acc, valid_log