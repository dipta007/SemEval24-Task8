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
        with jsonlines.open(f'./data/aug_data/SubtaskA/monolingual/ibm/{split}.jsonl') as reader:
            for obj in reader:
                obj['label'] = -1 if obj['label'] == HUMAN else 1
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
            self.test_dataset = self.get_dataset("test")
        elif stage == "test_final":
            data = []
            with jsonlines.open(f'./data/SubtaskA/subtaskA_test_monolingual.jsonl') as reader:
                for obj in reader:
                    obj['label'] = 0
                    obj['gen_text'] = ""
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
        text = [obj['text'] for obj in batch]
        gen_text = [obj['gen_text'] for obj in batch]
        label = [obj['label'] for obj in batch]
        ids = [obj['id'] for obj in batch]

        text = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        gen_text = self.tokenizer(gen_text, return_tensors='pt', padding=True, truncation=True)

        text['global_attention_mask'] = torch.zeros_like(text['input_ids'])
        text['global_attention_mask'][:, 0] = 1
        
        gen_text['global_attention_mask'] = torch.zeros_like(gen_text['input_ids'])
        gen_text['global_attention_mask'][:, 0] = 1

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