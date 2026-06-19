import torch

class DynamicLoss:
    def __init__(
        self,
        ema_alpha= 0.9,
        power_p= 1.5,
        eps= 1e-8,
        lambda_min= 0.1,
        lambda_max= 0.9
    ):
        super().__init__()
        
        self.ema_alpha = ema_alpha
        self.power_p = power_p
        self.eps = eps
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        
        self.reset()
    
    def reset(self):
        self.ema_c = None
        self.ema_t = None
        
        self.lambda_c_sum = 0.0
        self.lambda_t_sum = 0.0
        self.lambda_steps = 0
    
    def __call__(self, c_loss, t_loss):
        with torch.no_grad():
            c_val = c_loss.detach().item()
            t_val = t_loss.detach().item()
        
        if self.ema_c is None:
            self.ema_c = c_val
            self.ema_t = t_val
        else:
            self.ema_c = (
                self.ema_alpha * self.ema_c + (1.0 - self.ema_alpha) * c_val
            )
            self.ema_t = (
                self.ema_alpha * self.ema_t + (1.0 - self.ema_alpha) * t_val
            )
        
        wc = (self.ema_c + self.eps) ** self.power_p
        wt = (self.ema_t + self.eps) ** self.power_p
        total_weight = wc + wt
        
        lambda_c = wc / total_weight
        lambda_c = float(max(self.lambda_min, min(self.lambda_max, lambda_c)))
        lambda_t = 1.0 - lambda_c
        
        loss = lambda_c * c_loss + lambda_t * t_loss
        
        self.lambda_c_sum += lambda_c
        self.lambda_t_sum += lambda_t
        self.lambda_steps += 1
        
        return loss, lambda_c, lambda_t
    
    def get_lambda_mean(self):
        if self.lambda_steps == 0:
            return 0.5, 0.5
        
        lambda_c_mean = self.lambda_c_sum / self.lambda_steps
        lambda_t_mean = self.lambda_t_sum / self.lambda_steps
        
        return lambda_c_mean, lambda_t_mean