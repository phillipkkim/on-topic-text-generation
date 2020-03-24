import transformers
import gpt2.encoder
import gpt2.model
import gpt2.config
import gpt2.utils
import torch
import csv_dataset
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bert():
    '''
    @return bert tokenizer, bert model
    '''
    print('[getting pretrained bert tokenizer and model]')
    pretrained_name = 'bert-base-cased'
    tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_name)
    model = transformers.BertModel.from_pretrained(pretrained_name)
    model = model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


def get_gpt():
    '''
    @return gpt encoder, gpt trsnsformer (not head model, does not include lm model fc)
    '''
    print('[getting pretrained gpt2 tokenizer and model]')
    gpt_encoder = gpt2.encoder.get_encoder()

    state_dict = torch.load('gpt2/gpt2-pytorch_model.bin',
                            map_location='cpu' if not torch.cuda.is_available() else None)
    config = gpt2.config.GPT2Config()
    headmodel = gpt2.model.GPT2LMHeadModel(config)
    headmodel = gpt2.utils.load_weight(headmodel, state_dict)
    headmodel.to(DEVICE)
    headmodel.eval()
    return gpt_encoder, headmodel


def get_dataloaders(batch_size, codec):
    print('[getting dataloaders]')

    def path(filename):
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', 'data', filename)
    return [
        torch.utils.data.DataLoader(
            dataset=csv_dataset.NewsDataset(
                path=path(filename),
                ctx_length=128,
                codec=codec,
                start_from_zero=(False if 'train'in filename else True)
            ),
            batch_size=batch_size,
            shuffle=(True if 'train' in filename else False),
            num_workers=8
        )
        for filename in ['train.csv', 'dev.csv', 'test.csv']
    ]
