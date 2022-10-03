from numpy import squeeze
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
from torch.nn.modules.module import Module



class WeightBCEWithLogitsLoss(Module):
	"""
	BCEwithLogitsLoss with element-wise weight
	"""
	def __init__(self):
		super(WeightBCEWithLogitsLoss, self).__init__()
		self.bce = BCEWithLogitsLoss(reduction="none")

	def forward(self, inputs, target, weights):
		loss = self.bce(inputs, target)
		if weights is not None:
			loss = torch.mul(loss, weights)
		loss = torch.sum(loss)
		return loss

