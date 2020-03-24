import torch
import os
from transformers import BertModel, BertTokenizer


def bert_forward(model, tokenizer, highlights, device, padding=False):
    """
    @param model (BertModel): must be imported from transformers, not pytorch_transformers; initialized like "model = BertModel.from_pretrained('bert-base-uncased')"
    @param tokenizer (BertTokenizer): must be imported from transformers, not pytorch_transformers; initialized like "tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')"
    @param highlights (list of strings): list of highlights
    @param padding (bool)
    @return result (list of Tensors): tensor encoding for each highlight in highlights
    """
    model = model.to(device)
    model.eval()
    result = []
    with torch.no_grad():
        for i in range(len(highlights)):
            encoding = tokenizer.encode(
                highlights[i], max_length=512, add_special_tokens=False, pad_to_max_length=padding)
            input_ids = torch.tensor(encoding).unsqueeze(0).to(device)  # batch size is 1
            outputs = model(input_ids)
            # The last hidden-state is the first element of the output tuple
            last_hidden_states = outputs[0]
            last_hidden_states = torch.sum(last_hidden_states, dim = 1)/last_hidden_states.shape[0] # avgpool
            result.append(last_hidden_states)

    return torch.cat(result, dim = 0)


def bert_forward_single(model, tokenizer, highlight, device, padding=False):
    """
    @param model (BertModel): must be imported from transformers, not pytorch_transformers; initialized like "model = BertModel.from_pretrained('bert-base-uncased')"
    @param tokenizer (BertTokenizer): must be imported from transformers, not pytorch_transformers; initialized like "tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')"
    @param highlight (string)
    @param padding (bool)
    @return result (list of Tensors): tensor encoding for each highlight in highlights
    """
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        encoding = tokenizer.encode(
            highlight, max_length=512, add_special_tokens=False, pad_to_max_length=padding)
        input_ids = torch.tensor(encoding).unsqueeze(0).to(device)  # batch size is 1
        outputs = model(input_ids)
        # The last hidden-state is the first element of the output tuple
        last_hidden_states = outputs[0].squeeze(0)
        last_hidden_states = torch.sum(last_hidden_states, dim = 0)/last_hidden_states.shape[0] # avgpool
    return last_hidden_states


if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')

    # input_ids = torch.tensor(tokenizer.encode(
    #     "Hello, my dog is cute", add_special_tokens=False)).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids)

    # last_hidden_states = outputs[0]
    last_hidden_states = bert_forward(model, tokenizer, ["Hello, my dog is cute", "Hello, my dog is not cute"])
    print(last_hidden_states.shape)
