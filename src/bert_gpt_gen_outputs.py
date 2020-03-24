import numpy as np
import pandas as pd
import torch
import train_utils
import os
import bert_gpt_lm
import bert_gpt_dau
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

# CONFIG ###################################################
MODEL_GETTER = bert_gpt_dau.get_bert_gpt_dau_model
MODEL_TYPE_STR = 'bert_gpt_dau'
POOL_METHOD = 'max'
WITH_PROMPT = True 
SAMPLE_TEMPERATURE = 0.7
SAVE_EVERY_N_ROWS = 50
GENERATE_LENGTH = 128
############################################################

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_CSV_FILENAME = f'{MODEL_TYPE_STR}_{POOL_METHOD}_{"with_prompt" if WITH_PROMPT else "without_prompt"}.csv'
SAVE_FILEPATH = os.path.join(SCRIPT_DIR, '..', 'outputs', SAVE_CSV_FILENAME)

if __name__ == "__main__":
    print(f'[saving to {os.path.realpath(SAVE_FILEPATH)}]')

    head_model, bert_tokenizer, bert_model, gpt_encoder, gpt_model = \
        MODEL_GETTER(POOL_METHOD)
    _, _, test_data_loader = train_utils.get_dataloaders(
        batch_size=1, codec=gpt_encoder)

    generated_dict = {}
    latest_saved_idx = 0

    if os.path.exists(SAVE_FILEPATH):
        print(f'[resuming from csv]')
        df = pd.read_csv(SAVE_FILEPATH)
        generated_dict = df.to_dict(orient='index')
        latest_saved_idx = len(df) - 1
        pprint(generated_dict)
        print(f'[latest saved row {latest_saved_idx}]')

    for idx, data in enumerate(test_data_loader):
        print(f'{idx}.', end='', flush=True)
        if idx in generated_dict:
            continue

        highlight_str = data['highlight'][0]
        reference_str = gpt_encoder.decode(data['story'][0].tolist())

        if WITH_PROMPT:
            context_gpt_tokens = gpt_encoder.encode(highlight_str)
        else:
            context_gpt_tokens = [50256]  # <|endoftext|>
        context_gpt_tokens = torch.LongTensor(context_gpt_tokens)

        generated_gpt_tokens = head_model.generate(
            highlight_str=highlight_str,
            context_gpt_tokens=context_gpt_tokens,
            gen_length=GENERATE_LENGTH,
            use_sample=False,
            temperature=SAMPLE_TEMPERATURE
        )
        # trim off prompt
        generated_gpt_tokens = generated_gpt_tokens[len(
            context_gpt_tokens):]
        generated_str = gpt_encoder.decode(generated_gpt_tokens)
        generated_dict[idx] = {
            'generated': generated_str,
            'reference': reference_str,
            'highlight': highlight_str
        }

        if (idx+1) % SAVE_EVERY_N_ROWS == 0:
            print('[saving]')
            df = pd.DataFrame.from_dict(
                generated_dict,
                orient='index',
            )
            df.to_csv(SAVE_FILEPATH, index=False, header=True)

    print('[saving final]')
    df = pd.DataFrame.from_dict(
        generated_dict,
        orient='index',
    )
    df.to_csv(SAVE_FILEPATH, index=False, header=True)
