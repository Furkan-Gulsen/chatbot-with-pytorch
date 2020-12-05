import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open("intents.json", "r") as f:
	intents = json.load(f)

# print(intents)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
	tag = intent['tag']
	tags.append(tag)
	for pattern in intent['patterns']:
		w = tokenize(pattern)
		all_words.extend(w)
		xy.append((w, tag))

ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
	bag = bag_of_words(pattern_sentence, all_words)
	X_train.append(bag)

	label = tags.index(tag)
	y_train.append(label) # CrossEntropyLoss

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
	def __init__(selff):
		self.n_samples = len(X_train)
		self.x_data = X_train
		self.y_data = y_train

	def __getitem__(self, index):
		return self.x_data[idx], self.y_data[idx]

	def __len__(self):
		return self.n_samples

# Hyperparameters
batch_size = 8

dataset = ChatDataset()
train_loader = DataLoader(dataset= dataset, batch_size=batch_size, shuffle=True, num_workers=0)