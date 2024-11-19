import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r, lora_alpha, lora_dropout):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_A = nn.Parameter(torch.randn(original_linear.out_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, original_linear.in_features) * 0.01)
        self.scaling = lora_alpha / r

    def forward(self, x):
        result = self.original_linear(x)
        if self.r > 0:
            lora_result = self.lora_dropout(x) @ self.lora_B.t() @ self.lora_A.t()
            result += self.scaling * lora_result
        return result
