import torch

if __name__ == "__main__":
    cost_metrics = torch.load("cost_metrics.pt", weights_only=False)
    print(cost_metrics.size())