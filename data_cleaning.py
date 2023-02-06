import os
import pandas as pd

def clean_duplicates(config):


    dir_base = config["dir_base"]
    dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/second_labeled_batch.xlsx')
    #dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/first_labeled_batch.xlsx')

    #df = pd.concat(pd.read_excel(dataframe_location, sheet_name=None, engine='openpyxl'), ignore_index=True)
    #df = pd.read_excel(dataframe_location, engine='openpyxl')
    df = pd.read_excel(dataframe_location, sheet_name="Head CT", engine='openpyxl')
    #print(df)
    df = df.dropna(axis=0, how='all')
    #print(df)

    pd.set_option('display.max_columns', None)

    string_dic = {}
    dups = 0

    label_idx = 0
    data_with_labels = pd.DataFrame(columns=['id', 'impression1', 'impression2', 'label'])
    index = -1
    num_neg = 0
    discrepancy_that_are_nan = 0
    non_matching = 0
    prelim_impression = "string1"
    prelim_accession = "id1"
    final_impression = "string2"
    final_accession = "id2"
    prelim_num = 0
    final_num = 0
    num_same_string = 0


    for _, row in df.iterrows():

        if pd.isna(row['Accession Number']):
            continue
        if row["Report Type"] == "Preliminary":
            prelim_impression = row['Impression']
            prelim_accession = row['Accession Number']
            prelim_num += 1
        elif row["Report Type"] == "Final":
            final_impression = row['Impression']
            final_accession = row['Accession Number']
            final_num += 1
        index += 1
        #print(row['Discrepancy score'])
        #if pd.isna(row['Accession Number']):
        #    continue
        if pd.isna(row['Discrepancy']):
            discrepancy_that_are_nan += 1
            continue
        #print(row)
        if str(prelim_accession) == str(final_accession):

            if prelim_impression == final_impression:
                num_same_string += 1
                continue

            string_key = prelim_impression + final_impression
            if string_key in string_dic.keys():

                string_dic[string_key].append(prelim_accession)
                # df.drop(row["id"])
                dups += 1
                #print(f"label of dup: {label}")
                continue
            else:
                string_dic[string_key] = [prelim_accession]

            if row['Discrepancy'] == 0:
                label = 0
                if num_neg < 2800:
                    data_with_labels.loc[label_idx] = [prelim_accession, prelim_impression, final_impression, label]
                    num_neg += 1
            else:

                label = 1
                data_with_labels.loc[label_idx] = [prelim_accession, prelim_impression, final_impression, label]
            label_idx += 1
        else:
            non_matching += 1

    print(f"num unmatched: {non_matching}")
    print(f"discrepancy nans: {discrepancy_that_are_nan}")
    print(f"times prelim is defined: {prelim_num}")
    print(f"times final is defined: {final_num}")
    print(f"discrepancies delcared: {label_idx}")
    print(f"number of same strings: {num_same_string}")
    print(f"duplicates: {dups}")

    #remove_duplicate_strings(data_with_labels)
    return data_with_labels

def count_duplicates(config):

    dir_base = config["dir_base"]
    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/discrepancy_data/second_labeled_batch.xlsx')
    # dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/first_labeled_batch.xlsx')

    df = pd.concat(pd.read_excel(dataframe_location, sheet_name=None, engine='openpyxl'), ignore_index=True)
    # df = pd.read_excel(dataframe_location, engine='openpyxl')
    #df = pd.read_excel(dataframe_location, sheet_name="Head CT", engine='openpyxl')
    print(df)
    df = df.dropna(axis=0, how='all')
    print(df)

    pd.set_option('display.max_columns', None)

    string_dic = {}
    dups = 0

    label_idx = 0
    data_with_labels = pd.DataFrame(columns=['id', 'impression1', 'impression2', 'label'])
    index = -1
    num_neg = 0
    discrepancy_that_are_nan = 0
    non_matching = 0
    prelim_impression = "string1"
    prelim_accession = "id1"
    final_impression = "string2"
    final_accession = "id2"
    prelim_num = 0
    final_num = 0
    num_same_string = 0
    num_report_pairs = 0

    for _, row in df.iterrows():

        if pd.isna(row['Accession Number']):
            continue
        if row["Report Type"] == "Preliminary":
            prelim_impression = row['Impression']
            prelim_accession = row['Accession Number']
            prelim_num += 1
        elif row["Report Type"] == "Final":
            final_impression = row['Impression']
            final_accession = row['Accession Number']
            final_num += 1
        index += 1

        #if pd.isna(row['Discrepancy']):
        #    discrepancy_that_are_nan += 1
        #    continue
        # print(row)

        if str(prelim_accession) == str(final_accession):
            num_report_pairs += 1
            if str(prelim_impression) == str(final_impression):
                num_same_string += 1
                #continue
                #print(f"label of dup: {row['Discrepancy']}")

            string_key = str(prelim_impression) + str(final_impression)

            if string_key in string_dic.keys():

                string_dic[string_key].append(prelim_accession)
                # df.drop(row["id"])
                dups += 1
                #print(f"label of dup: {row['Discrepancy']}")
                continue
            else:
                string_dic[string_key] = [prelim_accession]

            #if row['Discrepancy'] == 0:
            #    label = 0
            #    if num_neg < 2800:
            #        data_with_labels.loc[label_idx] = [prelim_accession, prelim_impression, final_impression, label]
            #        num_neg += 1
            #else:

            #    label = 1
            #    data_with_labels.loc[label_idx] = [prelim_accession, prelim_impression, final_impression, label]
            #label_idx += 1
        else:
            non_matching += 1

    print(f"num unmatched: {non_matching}")
    print(f"discrepancy nans: {discrepancy_that_are_nan}")
    print(f"times prelim is defined: {prelim_num}")
    print(f"times final is defined: {final_num}")
    print(f"discrepancies delcared: {label_idx}")
    print(f"number of same strings: {num_same_string}")
    print(f"duplicates: {dups}")

    list_assention = string_dic.values()
    print(f"unique reports: {len(list_assention)}")

    num_duplicate_reports = 0
    for list in list_assention:
        if len(list) > 1:
            num_duplicate_reports += 1

    print(f"instances of dup reports: {num_duplicate_reports}")
    #print(string_dic.values())
    #print(type(string_dic.values()))
    # remove_duplicate_strings(data_with_labels)
    print(f"number of report pairs: {num_report_pairs}")

    return data_with_labels

def pick_test_set(config):

    dir_base = config["dir_base"]
    save_path = os.path.join(dir_base, 'Zach_Analysis/discrepancy_data/extended_labels_to_unlabled_examples.xlsx')
    df = extend_labeled_data(config)
    df.to_excel(save_path, index=False)
    #df = get_dataframe_with_unique_unlabeled_samples(config)
    #df.to_excel(save_path, index=False)

    print(df)

def extend_labeled_data(config):
    dir_base = config["dir_base"]
    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/discrepancy_data/second_labeled_batch_hand_cleaned.xlsx')
    # dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/first_labeled_batch.xlsx')

    df = pd.concat(pd.read_excel(dataframe_location, sheet_name=None, engine='openpyxl'), ignore_index=True)
    df = df.dropna(axis=0, how='all')
    pd.set_option('display.max_columns', None)

    string_dic = {}
    dups = 0

    labeled_data_location = os.path.join(dir_base, 'Zach_Analysis/discrepancy_data/second_set_matches_removed_hand_cleaned_df.xlsx')
    df_labeled = pd.read_excel(labeled_data_location, sheet_name=None, engine='openpyxl')
    print(df_labeled)
    print(type(df_labeled))

    for _, row in df_labeled.itterrows():
        # gets the two impressions and uses it as a key to store the label of the report pairs
        prelim_impression = row["impression1"]
        final_impression = row["impression2"]
        string_key = str(prelim_impression) + str(final_impression)
        if string_key in string_dic.keys():
            continue
        else:
            string_dic[string_key] = [row["label"]]


    data_with_labels_extended = pd.DataFrame(
        columns=["Accession Number", "Birth Date", "Study Date / Time", "Report Date / Time", "Procedure Description",
                 "Diagnosis", "Review", "Discrepancy", "Discrepancy score", "Report Type", "Impression", "Report Body"])

    non_matching = 0
    prelim_impression = "string1"
    prelim_accession = "id1"
    final_impression = "string2"
    final_accession = "id2"
    prelim_num = 0
    final_num = 0
    num_same_string = 0
    num_report_pairs = 0
    prelim_row = ""
    index = -1


    for _, row in df.iterrows():

        if pd.isna(row['Accession Number']):
            continue
        if row["Report Type"] == "Preliminary":
            prelim_impression = row['Impression']
            prelim_accession = row['Accession Number']
            prelim_row = row
            prelim_num += 1
        elif row["Report Type"] == "Final":
            final_impression = row['Impression']
            final_accession = row['Accession Number']
            final_num += 1

        if str(prelim_accession) == str(final_accession):

            num_report_pairs += 1
            # labels the report non discrepant if their is no difference between reports
            if str(prelim_impression) == str(final_impression):
                num_same_string += 1
                row["Discrepancy"] = 0

            # extends a label if it already is labeled
            string_key = str(prelim_impression) + str(final_impression)
            if string_key in string_dic.keys():
                row["Discrepancy"] = string_dic[string_key]

            index += 1
            data_with_labels_extended.loc[index] = prelim_row
            index += 1
            data_with_labels_extended.loc[index] = row
            index += 1
            empty_row = pd.Series([None, None, None, None, None, None, None, None, None, None, None, None])
            data_with_labels_extended.loc[index] = empty_row
        else:
            non_matching += 1

    print(f"num unmatched: {non_matching}")
    print(f"times prelim is defined: {prelim_num}")
    print(f"times final is defined: {final_num}")
    print(f"number of same strings: {num_same_string}")
    print(f"duplicates: {dups}")

    return data_with_labels_extended


def get_dataframe_with_unique_unlabeled_samples(config):
    dir_base = config["dir_base"]
    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/discrepancy_data/second_labeled_batch.xlsx')
    # dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/first_labeled_batch.xlsx')

    df = pd.concat(pd.read_excel(dataframe_location, sheet_name=None, engine='openpyxl'), ignore_index=True)
    # df = pd.read_excel(dataframe_location, engine='openpyxl')
    #df = pd.read_excel(dataframe_location, sheet_name="Head CT", engine='openpyxl')
    #print(df)
    df = df.dropna(axis=0, how='all')
    #print(df)

    pd.set_option('display.max_columns', None)

    string_dic = {}
    dups = 0

    data_without_labels = pd.DataFrame(columns=["Accession Number", "Birth Date", "Study Date / Time", "Report Date / Time", "Procedure Description", "Diagnosis", "Review", "Discrepancy", "Discrepancy score", "Report Type", "Impression", "Report Body"])

    non_matching = 0
    prelim_impression = "string1"
    prelim_accession = "id1"
    final_impression = "string2"
    final_accession = "id2"
    prelim_num = 0
    final_num = 0
    num_same_string = 0
    num_report_pairs = 0
    prelim_row = ""
    index = -1

    for _, row in df.iterrows():

        if pd.isna(row['Accession Number']):
            continue
        if row["Report Type"] == "Preliminary":
            prelim_impression = row['Impression']
            prelim_accession = row['Accession Number']
            prelim_row = row
            prelim_num += 1
        elif row["Report Type"] == "Final":
            final_impression = row['Impression']
            final_accession = row['Accession Number']
            final_num += 1

        if str(prelim_accession) == str(final_accession):

            num_report_pairs += 1
            if str(prelim_impression) == str(final_impression):
                num_same_string += 1
                continue

            string_key = str(prelim_impression) + str(final_impression)

            # skips the entry if it is already in the data
            if string_key in string_dic.keys():
                string_dic[string_key].append(prelim_accession)
                dups += 1
                continue
            else:
                string_dic[string_key] = [prelim_accession]

            if pd.isna(row['Discrepancy']):
                #data_without_labels = data_without_labels.append(prelim_row)
                #data_without_labels = data_without_labels.append(row)
                #empty_row = pd.DataFrame()
                #data_without_labels = data_without_labels.append(empty_row)
                index += 1
                data_without_labels.loc[index] = prelim_row
                index += 1
                data_without_labels.loc[index] = row
                index += 1
                #empty_row = pd.DataFrame(columns=["Accession Number", "Birth Date", "Study Date / Time", "Report Date / Time", "Procedure Description",
                # "Diagnosis", "Review", "Discrepancy", "Discrepancy score", "Report Type", "Impression", "Report Body"])
                empty_row = pd.Series([None, None, None, None, None, None, None, None, None, None, None, None])
                data_without_labels.loc[index] = empty_row
        else:
            non_matching += 1

    print(f"num unmatched: {non_matching}")
    print(f"times prelim is defined: {prelim_num}")
    print(f"times final is defined: {final_num}")
    print(f"number of same strings: {num_same_string}")
    print(f"duplicates: {dups}")

    return data_without_labels