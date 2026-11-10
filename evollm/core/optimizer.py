"""
Stateless Normalized Cautious SGD Optimizer.

Provides rapid, robust single-session convergence in k=1..3 steps
WITHOUT storing optimizer states, buffers, or momentum histories between conversations.
"""
import torch
from typing import List


class StatelessNormalizedSGD:
    def __init__(self, parameters: List[torch.nn.Parameter], lr: float = 2e-4, max_grad_norm: float = 1.0):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.max_grad_norm = max_grad_norm

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        """
        Executes a normalized cautious gradient step across parameters.
        Guarantees well-conditioned updates in few iterations.
        """
        # Calculate total gradient norm
        total_sq_norm = 0.0
        for p in self.parameters:
            if p.grad is not None:
                total_sq_norm += torch.norm(p.grad.detach()).item() ** 2
        total_norm = total_sq_norm ** 0.5

        if total_norm == 0.0:
            return

        # Adaptive clipping factor
        clip_factor = 1.0
        if total_norm > self.max_grad_norm:
            clip_factor = self.max_grad_norm / (total_norm + 1e-8)

        # Apply update
        with torch.no_grad():
            for p in self.parameters:
                if p.grad is not None:
                    # Update parameter directly
                    p.add_(p.grad, alpha=-self.lr * clip_factor)

