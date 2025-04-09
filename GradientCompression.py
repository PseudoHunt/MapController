!pip install transformers datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
from torch import nn
device = 'cuda'

# Your DynamicLoRALinear definition should be here

# Replace GPT-2 blocks
for block in model.transformer.h:
    block.mlp.c_fc = DynamicLoRALinear(768, 3072, rank=4, device=device).to(device)
    block.mlp.c_proj = DynamicLoRALinear(3072, 768, rank=4, device=device).to(device)

model.cuda()
model.train()
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

dataset = load_dataset("roneneldan/TinyStories", split="train[:0.5%]")

def tokenize(example):
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True).remove_columns(["text"])
dataset.set_format("torch")
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=default_data_collator)
import torch
import matplotlib.pyplot as plt

optimizer = torch.optim.Adam([], lr=1e-3)  # No trainable params
losses = []
step = 0
merge_every = 50

for epoch in range(1):
    for batch in loader:
        step += 1
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = input_ids.clone()

        # Forward pass (AB already injected)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()

        # Update all DynamicLoRALinear layers
        for name, module in model.named_modules():
            if isinstance(module, DynamicLoRALinear):
                if hasattr(module, "_input_cache"):
                    grad_matrix = module.A.grad @ module.B if module.A.grad is not None else torch.zeros_like(module.A @ module.B)
                    module.update_from_grad(grad_matrix)

        losses.append(loss.item())
        model.zero_grad()

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

        if step % merge_every == 0:
            print("Merging into base weights...")
            for m in model.modules():
                if isinstance(m, DynamicLoRALinear):
                    m.merge_into_base()
plt.plot(losses)
plt.title("Training Loss (Dynamic LoRA)")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
model.eval()
prompt = "Once upon a time"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0]))
class OjaCompressor:
    def __init__(self, in_dim, out_dim, rank=4, lr=1e-2, device='cuda'):
        self.rank = rank
        self.lr = lr
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.initialized = False

        # Buffers for U and V will be created after first gradient
        self.U = None
        self.V = None

    def update(self, grad):
        """
        grad: gradient matrix of shape (out_dim, in_dim)
        """
        if not self.initialized:
            # Initialize with truncated SVD on first gradient
            U, S, Vt = torch.linalg.svd(grad, full_matrices=False)
            U = U[:, :self.rank]
            S = S[:self.rank]
            Vt = Vt[:self.rank, :]
            sqrt_S = torch.diag(S.sqrt())

            self.U = U @ sqrt_S      # (out_dim, r)
            self.V = sqrt_S @ Vt     # (r, in_dim)
            self.initialized = True
        else:
            # Oja-style updates
            self.U += self.lr * (grad @ self.V.T - self.U @ (self.V @ self.V.T))
            self.V += self.lr * (self.U.T @ grad - self.V @ (self.U.T @ self.U))

    def get_lora_weights(self):
        return self.U, self.V

    def reset(self):
        self.initialized = False
        self.U = None
        self.V = None
compressor = OjaCompressor(in_dim=512, out_dim=512, rank=4)
compressor.update(grad_matrix)        # First time: SVD init
A, B = compressor.get_lora_weights()  # Use for AB injection
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, pca_lr=1e-2, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.device = device
        self.pca_lr = pca_lr

        # Frozen base layer
        self.linear = nn.Linear(in_features, out_features).to(device)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA-style buffers (A and B)
        self.register_buffer("A", torch.zeros(out_features, rank, device=device))
        self.register_buffer("B", torch.zeros(rank, in_features, device=device))

        # Compressor state
        self.reset_compressor()

    def forward(self, x):
        delta_w = self.A @ self.B
        return F.linear(x, self.linear.weight + delta_w, self.linear.bias)

    def update_from_grad(self, grad_matrix):
        """
        Updates AB using either SVD (first time) or Oja's rule (after).
        grad_matrix: shape (out_dim, in_dim)
        """
        if not self.initialized:
            # First step: use SVD
            U, S, Vt = torch.linalg.svd(grad_matrix, full_matrices=False)
            U = U[:, :self.rank]
            S = S[:self.rank]
            Vt = Vt[:self.rank, :]
            sqrt_S = torch.diag(S.sqrt())
            self.U = U @ sqrt_S
            self.V = sqrt_S @ Vt
            self.initialized = True
        else:
            # Oja update
            self.U += self.pca_lr * (grad_matrix @ self.V.T - self.U @ (self.V @ self.V.T))
            self.V += self.pca_lr * (self.U.T @ grad_matrix - self.V @ (self.U.T @ self.U))

        # Inject weights
        self.A.copy_(self.U)
        self.B.copy_(self.V)

    def merge_into_base(self):
        """
        Merge AB into the frozen base weight and reset LoRA and PCA state.
        """
        with torch.no_grad():
            self.linear.weight += self.A @ self.B
            self.A.zero_()
            self.B.zero_()
        self.reset_compressor()

    def reset_compressor(self):
        """
        Reset internal PCA state — will reinitialize using SVD next time.
        """
        self.initialized = False
        self.U = None
        self.V = None
# Create dynamic LoRA layer
dyn_lora = DynamicLoRALinear(in_features=512, out_features=512, rank=4).cuda()

# During training loop:
grad_matrix = some_gradient_tensor  # shape (out_dim, in_dim)
dyn_lora.update_from_grad(grad_matrix)

# Merge every N steps:
dyn_lora.merge_into_base()
