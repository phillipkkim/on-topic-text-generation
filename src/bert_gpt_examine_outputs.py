import pandas as pd
import os
import numpy as np

# WITH_PROMPT = True
OUTFILE_DIR = '../outputs'
# MODEL_PREFIX = 'bert_gpt_dau'
# FILENAME = f'{MODEL_PREFIX}_max_{"with_prompt" if WITH_PROMPT else "without_prompt"}.csv'
FILENAME = 'gpt2_with_prompt.csv'

# OUTFILE_DIR = '../outputs/bergpt_init_embed'
# FILENAME = 'newest_bergpt_test_noprompt.csv'
# FILENAME = 'newest_bergpt_test_prompt_new.csv'

# 520 1289 3416 2895 1484

if __name__ == "__main__":
    df_max = pd.read_csv(os.path.join(
        OUTFILE_DIR,  FILENAME
    ))

    try:
        while True:
            idx = np.random.randint(0, len(df_max) + 1)
            print(f'[{idx}]')

            print(f'[highlight]')
            print(df_max.iloc[idx].highlight)

            print(f'[max]')
            print(df_max.iloc[idx].generated)

            input()

    except KeyboardInterrupt:
        pass
