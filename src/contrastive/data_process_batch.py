import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import jsonlines
from pprint import pprint
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import os
import gc
import re
import time
import sys
import pickle


MODEL_NAME = 'ibm/qcpg-sentences'
RANDOM_SEED = 42
BATCH_SIZE = 256

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_all_lines(paragraph):
    paragraph = paragraph.replace("\n", " \n")
    curr_real = []
    for line in paragraph.split(". "):
        curr_real.extend(line.split("\n"))
    curr_real = [x.strip() for x in curr_real]
    curr_real = [x for x in curr_real if len(x) > 0]
    return curr_real

def get_paraphase_batch(lines):
    batch = tokenizer(lines, return_tensors='pt', padding=True, truncation=True).to(device)
    generated_ids = model.generate(batch['input_ids'], max_new_tokens=83)
    generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_sentence

def get_paraphase_whole(paragraph, real2para):
    paraphase = []
    paragraph = paragraph.replace("\n", " \n")
    for line in paragraph.split(". "):
        curr_real = line.split("\n")
        now_paraphase = [real2para[x.strip()] for x in curr_real]

        paraphase.append("\n".join(now_paraphase))

    paraphase = ". ".join(paraphase)
    return paraphase

def process_data():
    start = time.time()
    with jsonlines.open('../../data/SubtaskA/subtaskA_train_monolingual.jsonl') as reader:
        all_lines = []
        for obj in tqdm(reader, total=119757):
            all_lines.extend(get_all_lines(obj['text']))
        all_lines.sort(key=lambda x: len(x.split(" ")), reverse=True)

        all_paraphase = []
        for i in tqdm(range(0, len(all_lines), BATCH_SIZE)):
            now_paraphase = get_paraphase_batch(all_lines[i:i+BATCH_SIZE])
            all_paraphase.extend(now_paraphase)

        real2para = {real: gen for real, gen in zip(all_lines, all_paraphase)}
        real2para[""] = ""

        print(len(all_lines), len(all_paraphase), len(real2para))

        del all_lines, all_paraphase
        gc.collect()

        with open('./data/SubtaskA/real2para.pkl', 'wb') as f:
            pickle.dump(real2para, f)

    
    with jsonlines.open('../../data/SubtaskA/subtaskA_train_monolingual.jsonl') as reader:
        data = []
        for obj in tqdm(reader, total=119757):
            x = obj['text']
            obj['gen_text'] = get_paraphase_whole(x, real2para)
            data.append(obj)
        
        random.shuffle(data)
        Y = [obj['label'] for obj in data]
        train, val = train_test_split(data, test_size=0.04, random_state=RANDOM_SEED, stratify=Y)
        print("Train size: ", len(train))
        print("Val size: ", len(val))

        os.makedirs('./data/SubtaskA', exist_ok=True)
        with jsonlines.open('./data/SubtaskA/subtaskA_train_monolingual_gen.jsonl', mode='w') as writer:
            writer.write_all(train)
        with jsonlines.open('./data/SubtaskA/subtaskA_val_monolingual_gen.jsonl', mode='w') as writer:
            writer.write_all(val)
            
    print(time.time() - start)


if __name__ == "__main__":
    process_data()