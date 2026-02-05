# Drifting Models
After a morning of reading through the paper, I really like this formulation.

Minimal implementation of [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) (Deng et al., 2026).

One-step generator trained by evolving the pushforward distribution at training time. A drifting field V points generated samples toward data; V => 0 at equilibrium.

Instead of iterating at inference (diffusion/flow), iteration happens at training time. An anti-symmetric drifting field V = V⁺ - V⁻ attracts generated samples toward data and repels from other generated samples via a mean-shift kernel. The loss is just MSE(x, stopgrad(x + V)) no score functions, no ODE solvers, no noise schedules. At inference: one forward pass.

```
python drifting.py
```

Trains on 8 Gaussians and checkerboard. Outputs scatter plots and drift field visualizations.
