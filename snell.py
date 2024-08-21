import torch

class SnellOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.learning_rate = lr
        self.prev = None
        self.eps = 1e-8

    def step(self):
        with torch.no_grad():
            _tmp = self.params
            for i, param in enumerate(self.params):
                if param.grad is not None:
                    grad = param.grad
                    nvectors = param.grad.size(0)
                    if self.prev is not None and len(grad.shape)==2:
                        vs = []
                        for j in range(nvectors):
                            new_v = self.snell(grad[j], self.prev[i].grad[j])
                            vs.append(new_v)
                        new_dir = torch.stack(vs)
                        if new_dir.isnan().any():
                            raise
                        param -= self.learning_rate * new_dir
                    else:
                        param -= self.learning_rate * param.grad
                    
                    self.prev = _tmp

    def snell(self, cur:torch.Tensor, prev:torch.Tensor):
        with torch.no_grad():
            if (1-cur).any(): return cur
            norm_cur, norm_prev = torch.norm(cur), torch.norm(prev)
            cos = torch.dot(cur, prev) / (norm_cur * norm_prev + self.eps)
            sin1 = torch.sqrt(1-cos**2).clip(0, 1)
            cross = torch.cross(cur, prev)
            vel_cur, vel_prev = torch.exp(-norm_cur), torch.exp(-norm_prev)

            u_cur = cur / norm_cur
            u_cross = cross / torch.norm(cross)
            u_ortho = torch.cross(u_cur, u_cross)
            # new basis for x, y, z coord
            basis = torch.stack([u_cur, u_ortho, u_cross])

            # snell's law
            sin2 = sin1 / (vel_prev+self.eps) * vel_cur
            # prevent 전반사
            sin2 = sin2.clip(0, 1)
            cos2 = torch.sqrt(1-sin2**2).clip(0, 1)

            new_dir = torch.tensor([cos2, sin2, 0])
            return basis @ new_dir * norm_cur

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
