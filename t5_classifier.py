import torch


class T5Classifier(torch.nn.Module):
    def __init__(self, model, n_class = 1):
        super(T5Classifier, self).__init__()
        self.lang_encoder = model
        self.classifier = torch.nn.Linear(1024, n_class)

    def forward(self, ids, mask, token_type_ids):

        # feed text through t5 then average across encoding dimension and then do two class classification
        encoder_output = self.lang_encoder.encoder(input_ids=ids, attention_mask=mask, return_dict=True)
        pooled_sentence = encoder_output.last_hidden_state
        lang_rep_avg = torch.mean(pooled_sentence, 1)
        output = self.classifier(lang_rep_avg)

        return output