import argparse
import numpy as np
import torch
from transformers import BertForTokenClassification, BertTokenizerFast, AutoTokenizer
from transformers import LongformerForTokenClassification, LongformerTokenizerFast
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from utils import trim_entity_spans, convert_goldparse, ResumeDataset, tag2idx, idx2tag, get_hyperparameters, train_and_val_model


parser = argparse.ArgumentParser(description='Train Bert-NER')
parser.add_argument('-e', type=int, default=5, help='number of epochs')
parser.add_argument('-o', type=str, default='.',
                    help='output path to save model state')


args = parser.parse_args().__dict__

output_path = args['o']

MAX_LEN = 4096
EPOCHS = args['e']
MAX_GRAD_NORM = 1.0
MODEL_NAME = 'allenai/longformer-base-4096'
TOKENIZER = LongformerTokenizerFast(pretrained_model_name_or_path = 'allenai/longformer-base-4096', lowercase=True)
#TOKENIZER = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = 'allenai/longformer-base-4096', use_fast=True, lowercase=True)
TOKENIZER.add_tokens(["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"])
#TOKENIZER = BertTokenizerFast('./vocab/vocab.txt', lowercase=True)
DEVICE = torch.device("cuda")
print(DEVICE)
data = trim_entity_spans(convert_goldparse('data/Resumes.json'))

total = len(data)
train_data, val_data = data[:180], data[180:]

train_d = ResumeDataset(train_data, TOKENIZER, tag2idx, MAX_LEN)
val_d = ResumeDataset(val_data, TOKENIZER, tag2idx, MAX_LEN)

train_sampler = RandomSampler(train_d)
train_dl = DataLoader(train_d, sampler=train_sampler, batch_size=8)

val_dl = DataLoader(val_d, batch_size=4)

#model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(tag2idx))
model = LongformerForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(tag2idx))

model.to(DEVICE)
optimizer_grouped_parameters = get_hyperparameters(model, True)
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

train_and_val_model(
    model,
    TOKENIZER,
    optimizer,
    EPOCHS,
    idx2tag,
    tag2idx,
    MAX_GRAD_NORM,
    DEVICE,
    train_dl,
    val_dl
)

torch.save(
    {
        "model_state_dict": model.state_dict()
    },
    f'{output_path}/model-state.bin',
)
