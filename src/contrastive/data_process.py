import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import jsonlines
from pprint import pprint
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import os


MODEL_NAME = 'ibm/qcpg-sentences'
RANDOM_SEED = 42

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_paraphase(lines):
    batch = tokenizer(lines, return_tensors='pt', padding=True).to(device)
    generated_ids = model.generate(batch['input_ids'], max_new_tokens=83)
    generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_sentence


def get_paraphase_whole(paragraph):
    paraphase = []
    for line in paragraph.split(". "):
        curr_real = line.split("\n")
        now_paraphase = get_paraphase(curr_real)
        now_paraphase = [y_hat if len(y) > 0 else '' for y, y_hat in zip(curr_real, now_paraphase)]

        paraphase.append("\n".join(now_paraphase))

    paraphase = ". ".join(paraphase)
    return paraphase

def process_data():
    with jsonlines.open('../../data/SubtaskA/subtaskA_train_monolingual.jsonl') as reader:
        data = []
        for obj in tqdm(reader, total=119757):
            x = obj['text']
            obj['gen_text'] = get_paraphase_whole(x)
            data.append(obj)
        
        random.shuffle(data)
        Y = [obj['label'] for obj in data]
        train, val = train_test_split(data, test_size=0.5, random_state=RANDOM_SEED, stratify=Y)
        print("Train size: ", len(train))
        print("Val size: ", len(val))

        os.makedirs('./data/SubtaskA', exist_ok=True)
        with jsonlines.open('./data/SubtaskA/subtaskA_train_monolingual_gen.jsonl', mode='w') as writer:
            writer.write_all(train)
        with jsonlines.open('./data/SubtaskA/subtaskA_val_monolingual_gen.jsonl', mode='w') as writer:
            writer.write_all(val)


if __name__ == "__main__":
    process_data()