import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from enum import Enum
from tqdm import tqdm
import jsonlines
import torch
from collections import defaultdict
import random


MODELS = Enum("MODELS", ['human', 'chatGPT', 'cohere', 'davinci', 'bloomz', 'dolly'], start=0)

class ContrastiveDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()  
        self.config = config
        self.current_epoch = 0
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
    def get_dataset(self, split):
        seed_data = defaultdict(list)
        with jsonlines.open(f'./data/SubtaskB/subtaskB_{split}.jsonl') as reader:
            for obj in reader:
                name = MODELS(int(obj['label'])).name
                seed_data[name].append(obj)

        return seed_data
    
    def get_formatted_data(self, seed_data):
        for k in seed_data.keys():
            random.shuffle(seed_data[k])
        
        mn = min([len(seed_data[k]) for k in seed_data.keys()])
        data = []
        for i in range(mn):
            curr_data = []
            for k in MODELS:
                curr_data.append(seed_data[k.name][i])
            data.append(curr_data)
        
        return data

    def prepare_data(self):
        # download, tokenize, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage='fit'):
        if stage == "fit":
            self.train_dataset = self.get_dataset("train")
            self.val_dataset = self.get_dataset('dev_split10')
            self.formatted_val_dataset = self.get_formatted_data(self.val_dataset)
        elif stage == "test":
            data = []
            with jsonlines.open(f'./data/SubtaskB/subtaskB_dev.jsonl') as reader:
                for obj in reader:
                    curr_data = []
                    for _ in range(6):
                        curr_data.append(obj)
                    data.append(curr_data)
            self.test_dataset = data
        elif stage == "test_final":
            data = []
            with jsonlines.open(f'./data/test_final/subtaskB.jsonl') as reader:
                for obj in reader:
                    curr_data = []
                    for _ in range(6):
                        curr_data.append(obj)
                    data.append(curr_data)
            self.test_dataset = data
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        print(f"**** Epoch {self.current_epoch}: (Re)Loading train data ****")
        self.current_epoch += 1
        formatted_data = self.get_formatted_data(self.train_dataset)
        return DataLoader(formatted_data, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=1)
    
    def val_dataloader(self):
        return DataLoader(self.formatted_val_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=1)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=1)
    
    def collate_fn(self, batch):
        text = []
        ids = []
        for i in range(len(batch[0])):
            curr_text = []
            curr_ids = []
            for j in range(len(batch)):
                curr_text.append(batch[j][i]['text'])
                curr_ids.append(batch[j][i]['id'])
            text.append(curr_text)
            ids.append(curr_ids)

        for i in range(len(text)):
            text[i] = self.tokenizer(text[i], return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_sen_len)

            text[i]['global_attention_mask'] = torch.zeros_like(text[i]['input_ids'])
            text[i]['global_attention_mask'][:, 0] = 1

        for i in range(len(ids)):
            ids[i] = torch.tensor(ids[i], dtype=torch.int32)

        return text, ids


if __name__ == "__main__":
    from base_config import get_config
    config = get_config()
    datamodule = ContrastiveDataModule(config)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    for batch in datamodule.train_dataloader():
        data, ids = batch
        pos = data[0]
        neg = data[1]
        pos_ids = ids[0]
        neg_ids = ids[1]
        print(pos['input_ids'].shape)
        print(neg['input_ids'].shape)
        print(pos_ids.shape)
        print(neg_ids.shape)
        break