import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import jsonlines
import torch


HUMAN = 0
AI    = 1

class ContrastiveDocDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()  
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
    def get_dataset(self, split):
        data = []
        with jsonlines.open(f'./data/SubtaskA/subtaskA_{split}_monolingual_gen.jsonl') as reader:
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
            self.test_dataset = self.get_dataset("dev")
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=1)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=1)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=1)
    
    def get_token_and_attention_mask(self, docs):
        def get_sens(doc):
            sens = []
            doc = doc.replace("\n", " \n")
            for line in doc.split(". "):
                csens = line.split("\n")
                sens.extend(csens)

            sens = [x.strip() for x in sens]
            sens = [x for x in sens if len(x) > 0]
            sens = sens[:self.config.max_doc_len]

            return sens
        
        def get_padded_docs(docs):
            docs = [get_sens(doc) for doc in docs]
            docs_len = [len(doc) for doc in docs]

            max_len = max(docs_len)
            docs = [doc + [""] * (max_len - len(doc)) for doc in docs]

            # flatten docs
            docs = [x for doc in docs for x in doc]
            
            return docs, docs_len
        
        def get_attention_mask(docs_len):
            max_len = max(docs_len)
            doc_attn_mask = [[1] * doc_len + [0] * (max_len - doc_len) for doc_len in docs_len]
            return doc_attn_mask

        docs, docs_len = get_padded_docs(docs)
        doc_attn_mask = get_attention_mask(docs_len)
        return docs, doc_attn_mask

    
    def collate_fn(self, batch):
        text = [obj['text'] for obj in batch]
        gen_text = [obj['gen_text'] for obj in batch]
        label = [obj['label'] for obj in batch]
        ids = [obj['id'] for obj in batch]
        
        text, text_attn_mask = self.get_token_and_attention_mask(text)
        gen_text, gen_text_attn_mask = self.get_token_and_attention_mask(gen_text)

        text = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_sen_len)
        gen_text = self.tokenizer(gen_text, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_sen_len)
        label = torch.tensor(label, dtype=torch.float32)
        ids = torch.tensor(ids, dtype=torch.int32)

        text['doc_attention_mask'] = torch.tensor(text_attn_mask, dtype=torch.float32)
        gen_text['doc_attention_mask'] = torch.tensor(gen_text_attn_mask, dtype=torch.float32)

        return text, gen_text, label, ids


if __name__ == "__main__":
    from base_config import get_config
    config = get_config()
    datamodule = ContrastiveDocDataModule(config)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    for batch in datamodule.train_dataloader():
        text, gen_text, label, ids = batch
        print("text")
        print(text['input_ids'].shape)
        print(text['attention_mask'].shape)
        print(text['doc_attention_mask'].shape)
        print()

        print("gen")
        print(gen_text['input_ids'].shape)
        print(gen_text['attention_mask'].shape)
        print(gen_text['doc_attention_mask'].shape)
        print()

        print("target")
        print(label.shape)
        print(ids.shape)
        print()
        break