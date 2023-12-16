import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import jsonlines
import torch


HUMAN = 0
AI    = 1

class ContrastiveDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()  
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
    def get_dataset(self, split):
        data = []
        with jsonlines.open(f'./data/SubtaskA/subtaskA_{split}_monolingual_gen.jsonl') as reader:
            for obj in reader:
                obj['label'] = -1 if obj['label'] == 'HUMAN' else 1
                obj['gen_text'] = obj['gen_text'] if "gen_text" in obj else ""
                data.append(obj)
            data.append(obj)
        return data

    def prepare_data(self):
        # download, tokenize, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage='fit'):
        if stage == "fit":
            self.train_dataset = self.get_dataset("train")
            self.val_dataset = self.get_dataset("val")
        elif stage == "test":
            self.test_dataset = self.get_dataset("dev")
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate_fn)
    
    def collate_fn(self, batch):
        text = [obj['text'] for obj in batch]
        gen_text = [obj['gen_text'] for obj in batch]
        label = [obj['label'] for obj in batch]
        ids = [obj['id'] for obj in batch]

        text = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        gen_text = self.tokenizer(gen_text, return_tensors='pt', padding=True, truncation=True)
        label = torch.tensor(label, dtype=torch.float32)
        ids = torch.tensor(ids, dtype=torch.int32)

        return text, gen_text, label, ids


if __name__ == "__main__":
    from base_config import get_config
    config = get_config()
    datamodule = ContrastiveDataModule(config)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    for batch in datamodule.train_dataloader():
        text, gen_text, label, ids = batch
        print(text['input_ids'].shape)
        print(gen_text['input_ids'].shape)
        print(label.shape)
        print(ids.shape)
        break