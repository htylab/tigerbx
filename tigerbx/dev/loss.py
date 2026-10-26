import torch
import torch.nn as nn

class TopKLoss(nn.Module):
    def __init__(self, k):
        super(TopKLoss, self).__init__()
        self.k = k
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # You can use any other loss function here

    def forward(self, outputs, targets):
        # Compute the individual losses for each sample in the batch
        losses = self.criterion(outputs, targets)
        # Sort the losses in descending order and select the top-k losses
        topk_losses, _ = torch.topk(losses, self.k, dim=0, largest=True, sorted=True)
        # Compute the average of the top-k losses
        topk_loss = topk_losses.mean()
        return topk_loss

def test_TopKLoss():
    # Example usage
    outputs = torch.randn(32, 10)  # Example output from a neural network (batch_size=32, num_classes=10)
    targets = torch.randint(0, 10, (32,))  # Example ground truth labels

    k = 5  # Number of top-k samples to consider
    criterion = TopKLoss(k)

    loss = criterion(outputs, targets)
    print("Top-k Loss:", loss.item())
