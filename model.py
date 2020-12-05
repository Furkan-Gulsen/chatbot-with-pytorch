import torch
import torch.nn as nn

class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		self.l1 = nn.Linear(input_size, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(input_size, num_classes)
		self.relu = nn.ReLU()

	def forward(self, x):
		# first layer
		out = self.n1(x)
		out = self.relu(out)
		# second layer
		out = self.n2(out)
		out = self.relu(out)
		# third layer
		out = self.n3(out)
		# no activation and no softmax
		return out