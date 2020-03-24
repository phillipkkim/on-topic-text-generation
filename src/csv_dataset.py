import torch
import pandas as pd
import numpy as np

DATA_FILENAME = '../data/train.csv'
DATA_LENGTH = 92578


def random_crop(text_list, crop_length, start_from_zero):
    # text_list = text.split(" ")
    length = len(text_list)
    # print(length, crop_length)
    assert length > crop_length, "Text is too short, length:{}, {}, text: {}".format(
        length, crop_length, text_list)
    if start_from_zero:
        start_index = 0
    else:
        start_index = np.random.randint(
            0, min(length - crop_length, 1024 - crop_length))
    return np.array(text_list[start_index: start_index + crop_length]), np.arange(start_index, start_index + crop_length)


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, path, chunksize, nsamples=DATA_LENGTH, prompt_length=3):
        self.path = path
        self.chunksize = chunksize
        self.prompt_length = prompt_length
        self.len = int(nsamples / self.chunksize) + \
            (1 if nsamples / self.chunksize != 0 else 0)

    def __getitem__(self, index):
        df = next(
            pd.read_csv(
                self.path,
                skiprows=index * self.chunksize + 1,  # +1, since we skip the header
                chunksize=self.chunksize,
                names=['story', 'highlights']
            )
        )
        rows = [
            [row.highlights, " ".join(row.story.split(" ")[:self.prompt_length]), " ".join(row.story.split(" ")[self.prompt_length:])] for _, row in df.iterrows()
        ]
        return rows

    def __len__(self):
        return self.len


class HighlightsDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path, names=['story', 'highlights'])[
            "highlights"]

    def __getitem__(self, index):
        return self.df.iloc[index]

    def __len__(self):
        return self.df.shape[0]


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, path, ctx_length, codec, start_from_zero):
        self.path = path
        self.df = pd.read_csv(self.path)
        self.ctx_length = ctx_length
        self.codec = codec
        self.start_from_zero = start_from_zero

    def __getitem__(self, index):
        data = self.df.iloc[index]
        story, position_ids = random_crop(
            self.codec.encode(data["story"]),
            self.ctx_length,
            self.start_from_zero
        )
        # return {"highlight": data["highlights"], "story": data["story"]}
        return {"highlight": data["highlights"], "story": story, "pos_ids": position_ids}

    def __len__(self):
        return self.df.shape[0]


if __name__ == "__main__":
    # loader = torch.utils.data.DataLoader(
    #     dataset=CSVDataset(
    #         DATA_FILENAME,
    #         chunksize=2000,
    #         nsamples=DATA_LENGTH
    #     ),
    #     batch_size=1,  # keep this fixed at 1, and use chunksize in dataset instead
    #     num_workers=8,  # try to match num cpu
    #     shuffle=True
    # )
    #
    # for batch_idx, data in enumerate(loader):
    #     print(np.array(data).shape)
    #     print(f'loaded batch: {batch_idx}')

    dataset = NewsDataset(DATA_FILENAME, 128)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,  # keep this fixed at 1, and use chunksize in dataset instead
        num_workers=8,  # try to match num cpu
        shuffle=True
    )

    min_length = 1000000000
    count = 0
    for batch_idx, data in enumerate(loader):
        story = data["story"][0]
        # print(data["story"])
        length = len(story.split(" "))
        if min_length > length:
            min_length = length
        if length != 128:
            count += 1
            # print(length)
        if batch_idx % 1000 == 0:
            print(data["pos_ids"].shape)
            print('loaded batch: {}'.format(batch_idx))
    print(min_length, count)
