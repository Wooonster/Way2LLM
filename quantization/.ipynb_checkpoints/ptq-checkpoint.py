import torch
from tqdm import tqdm

class Model(torch.nn.Module):
    def __init__(self, in_f, inter_f):
        super().__init__()
        self.l1 = torch.nn.Linear(in_f, inter_f, bias=False)
        self.a1 = torch.nn.SiLU()
        self.l2 = torch.nn.Linear(inter_f, 2, bias=False)

    def forward(self, x):
        return self.l2(self.a1(self.l1(x)))


weights = torch.tensor(
    [[1, 2], [2, 4], [4, 5]], dtype=torch.float32
)
torch.manual_seed(525)
training_features = torch.randn(12000, 3, dtype=torch.float32)  # x
training_labels = training_features @ weights  # y

test_x = torch.randn(5525, 3, dtype=torch.float32)
test_y = test_x @ weights

model = Model(3, 5)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

for i in tqdm(range(100)):
    preds = model(training_features)
    loss = torch.nn.functional.mse_loss(preds, training_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


model.eval()
with torch.no_grad():
    preds = model(test_x)
    loss = torch.nn.functional.mse_loss(preds, test_y)
    print(f"float32(not quant) model test loss: {loss.item():.3f}")


# quantize
model_int8 = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
with torch.no_grad():
    preds = model_int8(test_x)
    loss = torch.nn.functional.mse_loss(preds, test_y)
    print(f"int8(int8 quant) model test loss: {loss.item():.3f}")

print()
print("float32 model linear1 parameter:\n", model.l1.weight)
print()
print("int8 model linear1 parameter(int8):\n", torch.int_repr(model_int8.l1.weight()))
print()
print("int8 model linear1 parameter(direct output):\n", model_int8.l1.weight())
      