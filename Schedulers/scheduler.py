from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR


class NoOpScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict.get("last_epoch", 0)


def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )


def lambda_scheduler(optimizer, args):
    """LambdaLR with a constant factor of 1.0; learning rate stays fixed."""
    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def no_scheduler(optimizer, args):
    """Keep the optimizer learning rate fixed across all steps."""
    return NoOpScheduler(optimizer)


schedulers = {
    "cosine": cosine_scheduler,
    "step": step_scheduler,
    "lambda": lambda_scheduler,
    "none": no_scheduler,
}
