from torch.optim.lr_scheduler import _LRScheduler


class CustomDecayLR(_LRScheduler):
    def __init__(self, optimizer, max_epochs, last_epoch=-1):
        self.num_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate according to the formula."""
        current_epoch = self.last_epoch + 1  # Increment epoch count
        lr_decay = (1 - current_epoch / self.num_epochs) ** 0.9
        return [base_lr * lr_decay for base_lr in self.base_lrs]
