import os
import json
import torch
import bert_gpt_dau
import bert_gpt_lm
import train_utils
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = [  # name, getter, pool_method
    ('bert_gpt_lm_avg', bert_gpt_lm.get_bert_gpt_lm_model, 'avg'),
    ('bert_gpt_lm_max', bert_gpt_lm.get_bert_gpt_lm_model, 'max'),
    ('bert_gpt_dau_max', bert_gpt_dau.get_bert_gpt_dau_model, 'max'),
    ('bert_gpt_dau_avg', bert_gpt_dau.get_bert_gpt_dau_model, 'avg'),
]
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(SCRIPT_DIR, '..', 'outputs')


def eval_ppl(model, loader, gpt_encoder, use_prompt):
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    totalloss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f'{batch_idx}.', end='', flush=True)
            batch_size, context_len = batch['story'].shape
            assert batch_size == 1

            story = batch['story']
            if use_prompt:
                storystring = gpt_encoder.decode(story[0].tolist())
                storystring = batch['highlight'][0] + storystring
                story = torch.LongTensor([gpt_encoder.encode(storystring)])
                highlight_token_len = len(
                    gpt_encoder.encode(batch['highlight'][0]))

            logits = model.train_forward(
                batch['highlight'],
                story
            )

            # the first word for each story has no corresponding prediction, remove it
            if use_prompt:
                logits = logits[highlight_token_len:, :]
                targets = story[:, highlight_token_len +
                                1:].to(DEVICE).reshape(-1)
            else:
                targets = story[:, 1:].to(DEVICE).reshape(-1)

            loss = criterion(logits, targets).item() / batch_size / context_len
            totalloss += loss

    print()
    ppl = np.exp(totalloss / len(loader))
    return ppl


if __name__ == "__main__":

    for use_prompt in [True, False]:
        for name, getter, pool_method in CONFIG:
            filename = f'test_perplexity_{"with_prompt" if use_prompt else "without_prompt"}.txt'
            print(filename)

            head_model, bert_tokenizer, bert_model, gpt_encoder, gpt_model = \
                getter(pool_method)
            _, _, test_data_loader = train_utils.get_dataloaders(
                batch_size=1, codec=gpt_encoder)

            ppl = eval_ppl(
                head_model, test_data_loader,
                gpt_encoder, use_prompt)
            print(f'{name}, ppl={ppl}')
            with open(os.path.join(SAVE_PATH, name, filename), 'w+') as f:
                f.write(str(ppl))
