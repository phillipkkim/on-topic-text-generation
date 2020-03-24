import transformers
import torch
import bert_gpt_lm
import bert_gpt_dau
import os
import train_utils
from pprint import pprint


NUM_EPOCH = 10
BATCH_SIZE = 60
SAVE_EVERY_N_BATCH = 20
GEN_EVERY_N_BATCH = 60


# CONFIG ###################################################

#MODEL_TO_TRAIN = bert_gpt_lm.BERTGPT2LMHeadModel
MODEL_TO_TRAIN = bert_gpt_dau.BERTGPT2DAUModel

POOL_METHOD = 'max'
#POOL_METHOD = 'avg'

############################################################

if MODEL_TO_TRAIN == bert_gpt_lm.BERTGPT2LMHeadModel:
    PATH_PREFIX = 'bert_gpt_lm'
elif MODEL_TO_TRAIN == bert_gpt_dau.BERTGPT2DAUModel:
    PATH_PREFIX = 'bert_gpt_dau'
else:
    raise()

SAVE_MODEL_PATH = f'../saved_models/{PATH_PREFIX}_{POOL_METHOD}/'
NEWEST_MODEL_SAVE_PATH = os.path.join(SAVE_MODEL_PATH, 'newest.pth')
BEST_MODEL_SAVE_PATH = os.path.join(SAVE_MODEL_PATH, 'best.pth')

SAVE_OUTPUT_PATH = f'../outputs/{PATH_PREFIX}_{POOL_METHOD}/'
TRAIN_LOSS_SAVE_PATH = os.path.join(SAVE_OUTPUT_PATH, 'train_loss.csv')
DEV_LOSS_SAVE_PATH = os.path.join(SAVE_OUTPUT_PATH, 'dev_loss.csv')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    bert_tokenizer, bert_model = train_utils.get_bert()
    gpt_encoder, gpt_model = train_utils.get_gpt()

    bert_gpt_head_model = MODEL_TO_TRAIN(
        bert_tokenizer, bert_model, gpt_encoder, gpt_model, POOL_METHOD)

    epoch_begin = 0
    batch_begin = 0
    best_dev_loss = float('inf')
    if os.path.exists(NEWEST_MODEL_SAVE_PATH):
        print('[load lm head weights, continue train]')
        print(f'[load weights from {NEWEST_MODEL_SAVE_PATH}]')
        bert_gpt_head_model.lm_head.load_state_dict(
            torch.load(
                NEWEST_MODEL_SAVE_PATH,
                map_location=DEVICE
            ),
        )
        with open(TRAIN_LOSS_SAVE_PATH) as f:
            tok = f.readlines()[-1].split(',')
            epoch_begin = int(tok[0])
            batch_begin = int(tok[1])
        if os.path.exists(DEV_LOSS_SAVE_PATH):
            with open(DEV_LOSS_SAVE_PATH) as f:
                tok = f.readlines()[-1].strip().split(',')
                best_dev_loss = float(tok[1])

        print(
            f'[starting at epoch {epoch_begin}, batch {batch_begin}, best dev {best_dev_loss}]')

    trainloader, devloader, testloader = train_utils.get_dataloaders(
        BATCH_SIZE, gpt_encoder)

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.AdamW(
        bert_gpt_head_model.lm_head.parameters(),  # important, dont train gpt or bert
        weight_decay=0.005
    )
    for epoch_idx in range(epoch_begin, NUM_EPOCH):
        print(f'\n[epoch] {epoch_idx}')
        trainloss = 0
        for batch_idx, data_batch in enumerate(trainloader):
            if batch_idx < batch_begin:
                continue
            batch_begin = 0
            print(f'{batch_idx}.', end='', flush=True)
            optimizer.zero_grad()
            dau_logits = bert_gpt_head_model.train_forward(
                data_batch['highlight'],
                data_batch['story']
            )

            # the first word for each story has no corresponding prediction, remove it
            targets = data_batch['story'][:, 1:].to(DEVICE)
            targets = targets.reshape(-1)  # flatten

            loss = criterion(dau_logits, targets)
            trainloss += loss.item()
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % SAVE_EVERY_N_BATCH == 0:
                torch.save(
                    bert_gpt_head_model.lm_head.state_dict(),
                    NEWEST_MODEL_SAVE_PATH)

                append_line(
                    TRAIN_LOSS_SAVE_PATH,
                    f'{epoch_idx},{batch_idx+1},{trainloss}')
                print()
                print(f'[trainloss] {trainloss}')
                trainloss = 0

            if (batch_idx+1) % GEN_EVERY_N_BATCH == 0:
                highlight_str = data_batch['highlight'][0]
                context_gpt_tokens = data_batch['story'][0][:1]
                generated_tokens = bert_gpt_head_model.generate(
                    highlight_str, context_gpt_tokens, use_sample=True)
                print('=' * 80)
                print('[highlight]')
                print(highlight_str)
                print('[context]')
                print(gpt_encoder.decode(context_gpt_tokens.tolist()))
                print('[generated]')
                print(gpt_encoder.decode(
                    generated_tokens[len(context_gpt_tokens):]))
                print('=' * 80)

        # after training one epoch, save dev model if its better
        devloss = 0
        with torch.no_grad():
            print('[dev eval]')
            for batch_idx, data_batch in enumerate(devloader):
                print(f'{batch_idx}.', end='', flush=True)
                dau_logits = bert_gpt_head_model.train_forward(
                    data_batch['highlight'],
                    data_batch['story']
                )
                targets = data_batch['story'][:, 1:].to(DEVICE)
                targets = targets.reshape(-1)  # flatten
                loss = criterion(dau_logits, targets)
                devloss += loss.item()
            print()
        append_line(
            DEV_LOSS_SAVE_PATH,
            f'{epoch_idx},{devloss}'
        )
        if devloss < best_dev_loss:
            print('[devloss improved, save]')
            torch.save(
                bert_gpt_head_model.lm_head.state_dict(),
                BEST_MODEL_SAVE_PATH
            )
            best_dev_loss = devloss


def append_line(filename, content):
    with open(filename, 'a') as f:
        f.write(f'{content}\n')


if __name__ == "__main__":
    train()
