import torch


class RobertaSingleClassifier(torch.nn.Module):
    def __init__(self, lang_model1, n_class = 1):
        super(RobertaSingleClassifier, self).__init__()
        self.lang_encoder1 = lang_model1
        self.classifier = torch.nn.Linear(768, n_class)


    def forward(self, ids1, mask1, filler1, filler2,  token_type_ids1, filler3, ): #, token_type_ids):

        lang_output1 = self.lang_encoder1(ids1, mask1, token_type_ids1)
        word_rep1 = lang_output1[0]
        report_rep1 = lang_output1[1]
        both_lang_rep = report_rep1
        # feed text through t5 then average across encoding dimension and then do two class classification
        #encoder_output1 = self.lang_encoder.encoder(input_ids=ids1, attention_mask=mask1, return_dict=True)
        #pooled_sentence1 = encoder_output1.last_hidden_state
        #lang_rep_avg1 = torch.mean(pooled_sentence1, 1)
        #print(lang_rep_avg1.size())
        # feed text through t5 then average across encoding dimension and then do two class classification
        #encoder_output2 = self.lang_encoder.encoder(input_ids=ids2, attention_mask=mask2, return_dict=True)
        #pooled_sentence2 = encoder_output2.last_hidden_state
        #lang_rep_avg2 = torch.mean(pooled_sentence2, 1)
        #print(lang_rep_avg2.size())
        #both_lang_rep = torch.cat((lang_rep_avg1, lang_rep_avg2), dim=1)
        #print(both_lang_rep.size())
        #print(both_lang_rep.size())
        output = self.classifier(both_lang_rep)

        return output