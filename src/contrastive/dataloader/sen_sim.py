import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from enum import Enum
from tqdm import tqdm
import jsonlines
import torch
import random


MODELS = Enum("MODELS", ['human', 'chatGPT', 'cohere', 'davinci', 'bloomz', 'dolly'], start=0)

class ContrastiveDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()  
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
    def get_dataset(self, split):
        seed_data = []
        with jsonlines.open(f'./data/SubtaskB/subtaskB_{split}.jsonl') as reader:
            for obj in reader:
                seed_data.append(obj)


        if split == 'dev':
            random.shuffle(seed_data)
            seed_data = seed_data[:1000]

        if True or self.config.debug:
            random.shuffle(seed_data)
            seed_data = seed_data[:100]

            if split == 'dev':
                seed_data = seed_data[:10]

        
        data = []
        for o1 in tqdm(seed_data, desc=f"Generating {split} dataset"):
            for o2 in seed_data:
                if o1['id'] == o2['id']:
                    continue
                    
                if o1['label'] == o2['label']:
                    continue
                
                nb = {
                    "pos_id": o1['id'],
                    "neg_id": o2['id'],
                    "pos": o1['text'],
                    "neg": o2['text'],
                    "pos_label": o1['label'],
                    "neg_label": o2['label'],
                }

                data.append(nb)

        print(f"Total {split} data: {len(data)}")
        return data

    def prepare_data(self):
        # download, tokenize, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage='fit'):
        if stage == "fit":
            self.train_dataset = self.get_dataset("train")
            self.val_dataset = self.get_dataset('dev')
        elif stage == "test":
            self.test_dataset = self.get_dataset("all_test")
        elif stage == "test_final":
            data = []
            with jsonlines.open(f'./data/test_final/subtaskA_monolingual.jsonl') as reader:
                for obj in reader:
                    data.append(obj)
                data.append(obj)
            self.test_dataset = data
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=1)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=1)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=1)
    
    def collate_fn(self, batch):
        pos = [obj['pos'] for obj in batch]
        neg = [obj['neg'] for obj in batch]
        pos_label = None
        neg_label = None
        if "pos_label" in batch[0]:
            pos_label = [obj['pos_label'] for obj in batch]
            neg_label = [obj['neg_label'] for obj in batch]
        
        pos_ids = [obj['pos_id'] for obj in batch]
        neg_ids = [obj['neg_id'] for obj in batch]

        pos = self.tokenizer(pos, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_sen_len)
        neg = self.tokenizer(neg, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_sen_len)

        pos['global_attention_mask'] = torch.zeros_like(pos['input_ids'])
        pos['global_attention_mask'][:, 0] = 1
        
        neg['global_attention_mask'] = torch.zeros_like(neg['input_ids'])
        neg['global_attention_mask'][:, 0] = 1

        if "pos_label" in batch[0]:
            pos_label = torch.tensor(pos_label, dtype=torch.int32)
            neg_label = torch.tensor(neg_label, dtype=torch.int32)
            pos_label = pos_label.type(torch.LongTensor)
            neg_label = neg_label.type(torch.LongTensor)
        
        pos_ids = torch.tensor(pos_ids, dtype=torch.int32)
        neg_ids = torch.tensor(neg_ids, dtype=torch.int32)

        return pos, neg, pos_label, neg_label, pos_ids, neg_ids


if __name__ == "__main__":
    from base_config import get_config
    config = get_config()
    datamodule = ContrastiveDataModule(config)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    for batch in datamodule.train_dataloader():
        pos, neg, pos_label, neg_label, pos_ids, neg_ids = batch
        print(pos['input_ids'].shape)
        print(neg['input_ids'].shape)
        print(pos_label.shape)
        print(neg_label.shape)
        print(pos_ids.shape)
        print(neg_ids.shape)
        print(pos_label)
        print(neg_label)
        break