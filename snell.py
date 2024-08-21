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
                        # If is there any previous record, just step 
                        param -= self.learning_rate * param.grad
                    self.prev[i] = copy.deepcopy(grad)

    def snell(self, cur:torch.Tensor, prev:torch.Tensor):
        with torch.no_grad():
            if not cur.any() or not prev.any(): 
                return cur
            if torch.all(cur==prev) or (cur-prev).count_nonzero() <= 1:
                return cur
            norm_cur, norm_prev = torch.norm(cur)+self.eps, torch.norm(prev)+self.eps
            cos = torch.dot(cur, prev) / (norm_cur * norm_prev + self.eps)
            a = (1-cos**2).clip(0, 1)
            sin1 = torch.sqrt(a)
            cross = torch.cross(cur, prev)
            vel_cur, vel_prev = torch.exp(-norm_cur), torch.exp(-norm_prev)

            u_cur = cur / norm_cur
            u_cross = cross / torch.norm(cross)
            u_ortho = torch.cross(u_cur, u_cross)
            # new basis for x, y, z coord
            basis = torch.stack([u_cur, u_ortho, u_cross], dim=1)

            # snell's law
            sin2 = sin1 / (vel_prev+self.eps) * vel_cur
            # prevent reflecting all
            sin2 = sin2.clip(0, 1)
            cos2 = torch.sqrt(1-sin2**2).clip(0, 1)

            new_dir = torch.tensor([cos2, sin2, 0])
            return basis @ new_dir * norm_cur

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
