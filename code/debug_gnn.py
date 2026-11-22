import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"PyTorch version: {torch.__version__}")

print("\n── 1. Sanity check: can this exact model class learn a trivial toy task? ──")
# Trivial toy: 2 graphs, fully separable by node feature mean alone.
torch.manual_seed(0)

class SimpleGCN(nn.Module):
    def __init__(self, in_dim=2, hidden=8):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        h = F.relu(adj @ self.w1(x))
        h = self.dropout(h)
        h = F.relu(adj @ self.w2(h))
        g = h.mean(dim=0)
        return self.out(g)

# 10 easy graphs: class 0 has node features near 0, class 1 has node features near 5
graphs_toy, labels_toy = [], []
rng = np.random.RandomState(0)
for i in range(10):
    label = i % 2
    x = rng.randn(10, 2).astype(np.float32) + (5.0 if label == 1 else 0.0)
    adj = np.eye(10, dtype=np.float32)  # trivial identity adjacency
    graphs_toy.append((x, adj))
    labels_toy.append(label)

model = SimpleGCN()
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
model.train()
for epoch in range(100):
    opt.zero_grad()
    loss = 0.0
    for x, adj in zip([g[0] for g in graphs_toy], [g[1] for g in graphs_toy]):
        pass
    for i in range(10):
        xt = torch.tensor(graphs_toy[i][0])
        adjt = torch.tensor(graphs_toy[i][1])
        logit = model(xt, adjt)
        y = torch.tensor([float(labels_toy[i])])
        loss = loss + F.binary_cross_entropy_with_logits(logit, y)
    loss = loss / 10
    loss.backward()
    opt.step()

model.eval()
print("Toy task predictions after training (should separate ~0 vs ~1 by label):")
with torch.no_grad():
    for i in range(10):
        xt = torch.tensor(graphs_toy[i][0])
        adjt = torch.tensor(graphs_toy[i][1])
        prob = torch.sigmoid(model(xt, adjt)).item()
        print(f"  label={labels_toy[i]}  predicted_prob={prob:.4f}")

print("\nIf these probabilities do NOT separate by label (e.g. all ~0.5, or all")
print("identical, or NaN), the bug is in torch/model mechanics, not the real data.")
print("If they DO separate correctly, the model works fine and the bug is")
print("something specific to the real PaHaW graph construction/training loop.")
