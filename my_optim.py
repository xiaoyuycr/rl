import torch
from torch.optim import Optimizer

class AsyncRMSprop(Optimizer):
    def __init__(self, parms, lr=0.001,
                 alpha=0.99, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super(AsyncRMSprop, self).__init__(parms, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.IntTensor([0])
                state['square_avg'] = torch.zeros_like(p.data)
                state['step'].share_memory_()
                state['square_avg'].share_memory_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                square_avg, step = state['square_avg'], state['step']
                lr, alpha = group['lr'], group['alpha']
                state['step'] += 1
                if group['weight_decay']!=0:
                    grad = grad.add(group['weight_decay'], p.data)
                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)  #s = alpha * a + (1-alpha)g^2
                avg = square_avg.sqrt().add_(group['eps'])  # avg = sqrt(s+eps)~sqrt(s)+eps
                p.data.addcdiv_(-group['lr'], grad, avg)
        return  loss
