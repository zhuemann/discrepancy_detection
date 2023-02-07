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
        output = self.classifier(both_lang_rep)

        return output