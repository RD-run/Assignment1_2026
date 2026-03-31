from torch.optim.lr_scheduler import LRScheduler


class LambdaLR(LRScheduler):
    """Multiplies each param-group learning rate by a user function."""

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        factor = self.lr_lambda(t)
        return [base_lr * factor for base_lr in self.base_lrs]

    def state_dict(self):
        return {
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
            "_step_count": self._step_count,
            "_get_lr_called_within_step": self._get_lr_called_within_step,
            "_last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict):
        self.base_lrs = state_dict["base_lrs"]
        self.last_epoch = state_dict["last_epoch"]
        self._step_count = state_dict["_step_count"]
        self._get_lr_called_within_step = state_dict["_get_lr_called_within_step"]
        self._last_lr = state_dict["_last_lr"]
