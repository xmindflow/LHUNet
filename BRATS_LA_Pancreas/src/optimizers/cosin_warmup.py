from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, LinearLR


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    PyTorch learning rate scheduler that implements a linear warmup followed by
    a cosine annealing schedule.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Number of epochs for the linear warmup.
        max_epochs (int): Total number of epochs including warmup.
        warmup_start_lr (float): Starting learning rate for the warmup.
        eta_min (float, optional): Minimum learning rate for the cosine annealing phase.
                                   Default: 0.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr,
        eta_min=0,
        last_epoch=-1,
    ):
        self._validate_hyperparameters(
            optimizer, warmup_epochs, max_epochs, warmup_start_lr, eta_min, last_epoch
        )

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        # Initialize the warmup and annealing schedulers
        self._init_schedulers(optimizer, last_epoch)

        super().__init__(optimizer, last_epoch)

    def _init_schedulers(self, optimizer, last_epoch):
        """Initializes the linear and cosine annealing schedulers."""
        self.warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.warmup_start_lr / optimizer.defaults["lr"],
            total_iters=self.warmup_epochs,
            last_epoch=last_epoch,
        )
        self.after_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=self.eta_min,
            last_epoch=last_epoch,
        )

    def get_lr(self):
        """Returns the current learning rates."""
        if self.last_epoch < self.warmup_epochs:
            return self.warmup_scheduler.get_lr()
        else:
            return self.after_scheduler.get_lr()

    def step(self, epoch=None):
        """
        Updates the learning rate for the current epoch.

        Args:
            epoch (int, optional): Current epoch. Default: None.
        """
        if epoch is not None and epoch < 0:
            raise ValueError("Epoch number should be non-negative")

        self.last_epoch = epoch if epoch is not None else self.last_epoch + 1

        if self.last_epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.after_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
            self.after_scheduler.step()

    def _validate_hyperparameters(
        self, optimizer, warmup_epochs, max_epochs, warmup_start_lr, eta_min, last_epoch
    ):
        """Validates the hyperparameters for the scheduler."""
        assert (
            warmup_epochs < max_epochs
        ), "Warmup epochs must be smaller than max epochs"
        assert (
            warmup_start_lr < optimizer.defaults["lr"]
        ), "Warmup start LR must be smaller than the initial LR"
        assert (
            eta_min <= optimizer.defaults["lr"]
        ), "Minimum LR must be smaller than the initial LR"
        assert last_epoch < max_epochs, "Last epoch must be smaller than max epochs"
        assert (
            last_epoch < warmup_epochs or last_epoch == -1
        ), "Last epoch must be smaller than or equal to warmup epochs when last_epoch is not default"
        assert warmup_epochs > 0, "Warmup epochs must be greater than 0"
        assert max_epochs > 0, "Max epochs must be greater than 0"
        assert warmup_start_lr > 0, "Warmup start LR must be greater than 0"
        assert eta_min >= 0, "Minimum LR must be greater than 0"
