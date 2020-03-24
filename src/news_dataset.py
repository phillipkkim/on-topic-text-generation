import torch
import pandas as pd
import numpy as np
import os, os.path

DATA_FILENAME = '../data/test'
GEN_DATA_FILENAME = '../output/original_gpt2'


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, path, prompt_length = 10, transformation = None):
        self.path = path
        self.prompt_length = prompt_length
        # print(os.path.join(self.path, "highlights"))
        # print([name for name in os.listdir(os.path.join(self.path, "highlights"))])
        self.len = len([name for name in os.listdir(os.path.join(self.path, "highlights"))])
        # print(self.len)

    def __getitem__(self, index):
        highlight_file = open(os.path.join(*[self.path, "highlights", str(index) + ".txt"]),"r")
        story_file = open(os.path.join(*[self.path, "stories", str(index) + ".txt"]),"r")
        highlight = highlight_file.read()
        story = story_file.read()
        prompt = " ".join(story.split(" ")[:self.prompt_length])
        target = " ".join(story.split(" ")[self.prompt_length:])
        return {"highlight": highlight, "prompt": prompt, "target": target}

    def __len__(self):
        return self.len


class NewsDatasetEval(torch.utils.data.Dataset):
    def __init__(self, path, gen_path, prompt_length = 10, transformation = None):
        self.path = path
        self.gen_path = gen_path
        self.prompt_length = prompt_length
        # print(os.path.join(self.path, "highlights"))
        # print([name for name in os.listdir(os.path.join(self.path, "highlights"))])
        self.len = len([name for name in os.listdir(os.path.join(self.path, "highlights"))])
        # print(self.len)

    def __getitem__(self, index):
        highlight_file = open(os.path.join(*[self.path, "highlights", str(index) + ".txt"]),"r")
        story_file = open(os.path.join(*[self.path, "stories", str(index) + ".txt"]),"r")
        gen_file = open(os.path.join(*[self.gen_path, str(index) + ".txt"]),"r")
        highlight = highlight_file.read()
        story = story_file.read()
        gen_story = gen_file.read()
        return {"highlight": highlight, "target": story, "gen": gen_story}

    def __len__(self):
        return self.len



if __name__ == "__main__":
    loader = torch.utils.data.DataLoader(
        dataset=NewsDatasetEval(
            DATA_FILENAME
        ),
        batch_size=2,  # keep this fixed at 1, and use chunksize in dataset instead
        num_workers=8,  # try to match num cpu
        shuffle=True
    )

    for batch_idx, data in enumerate(loader):
        if batch_idx == 0:
            print(data)
        # print(np.array(data).shape)
        # print(f'loaded batch: {batch_idx}')
