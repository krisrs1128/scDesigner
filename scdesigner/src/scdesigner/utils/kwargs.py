def _filter_kwargs(kwargs: dict, allowed: set) -> dict:
    """Return a new dict containing only keys from kwargs that are in allowed."""
    return {k: v for k, v in kwargs.items() if k in allowed}


# allowed kwargs for common components
DEFAULT_ALLOWED_KWARGS = {
    'trainer': {
        'max_epochs', 'gpus', 'devices', 'accelerator', 'precision',
        'logger', 'callbacks', 'strategy', 'num_nodes', 'limit_train_batches',
        'log_every_n_steps', 'accumulate_grad_batches'
    },
    'data': {
        'chunk_size', 'batch_size', 'shuffle', 'num_workers', 'device'
    },
    'optimizer': {
        'lr', 'learning_rate', 'momentum', 'weight_decay', 'eps', 'betas',
        'amsgrad', 'dampening', 'nesterov', 'alpha',
        'T_max', 'eta_min', 'step_size', 'gamma', 'milestones', 'last_epoch',
        'verbose', 'patience', 'threshold', 'cooldown',
        'optimizer_class', 'optimizer', 'scheduler_class', 'scheduler',
        'monitor', 'interval', 'frequency'
    },
    'initialize': {
        'tol', 'clip', 'max_iter'
    }
}