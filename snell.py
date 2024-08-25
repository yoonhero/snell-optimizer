import torch
import copy

class SnellOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.learning_rate = lr
        self.prev = list([None for i in range(len(self.params))])
        self.eps = 1e-8

    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is not None:
                    grad = param.grad
                    nvectors = param.grad.size(0)
                    if self.prev[i] is not None and len(grad.shape)==2:
                        vs = []
                        for j in range(nvectors):
                            new_v = self.snell(grad[j], self.prev[i][j])
                            vs.append(new_v)
                        new_dir = torch.stack(vs)
                        param -= self.learning_rate * new_dir
                    else:
                        # If there isn't any previous record, just step 
                        param -= self.learning_rate * param.grad
                    self.prev[i] = copy.deepcopy(grad)

    def snell(self, cur:torch.Tensor, prev:torch.Tensor):
        with torch.no_grad():
            if not cur.any() or not prev.any(): 
                return cur
            if torch.all(cur==prev) or (cur-prev).count_nonzero() <= 1:
                return cur
            norm_cur = torch.norm(cur).clamp(min=self.eps)
            norm_prev = torch.norm(prev).clamp(min=self.eps)
            cos = torch.clamp(torch.dot(cur, prev) / (norm_cur * norm_prev), min=-1, max=1)
            sin1 = torch.sqrt((1-cos**2).clamp(min=0, max=1))
            cross = torch.cross(cur, prev, dim=0)
            vel_cur, vel_prev = torch.exp(-norm_cur), torch.exp(-norm_prev)

            u_cur = cur / norm_cur
            cross_norm = torch.norm(cross).clamp(min=self.eps)
            u_cross = cross / cross_norm
            u_ortho = torch.cross(u_cur, u_cross)
            basis = torch.stack([u_cur, u_ortho, u_cross], dim=1)

            sin2 = torch.clamp(sin1 * vel_cur / vel_prev, max=1)
            cos2 = torch.sqrt(1 - sin2**2)

            new_dir = torch.stack([cos2, sin2, torch.tensor(0.0)])
            return basis @ new_dir * norm_cur

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
