import torch
from torchmetrics.classification import Accuracy

# Create a tensor of length 10 with 0s and 1s
tensor1 = torch.randint(0, 3, (10,))
tensor2 = torch.randint(0, 3, (10,))

accuracy_metric = Accuracy(task="multiclass", num_classes=3)
accuracy_metric.update(tensor1, tensor2)

train_acc = accuracy_metric.compute().item()

print(tensor1)
print(tensor2)
print(train_acc)