import types
import typing
import random
import logging
import tqdm

try:
    import timm
    import timm.optim
    import timm.scheduler
except ImportError:
    timm = None  # type: ignore
try:
    import torch
except ImportError:
    torch = None  # type: ignore
import numpy as np

LOGGER = logging.getLogger()
DEFAULT_SCHEDULER_PARAMS = dict(
    sched="cosine",
    min_lr=1e-5,
    decay_rate=1,
    warmup_lr=0,
    warmup_epochs=0,
    cooldown_epochs=0,
    epochs=100,
    lr_cycle_limit=0,
)

DEFAULT_OPTIMIZER_PARAMS = dict(learning_rate=1e-2, weight_decay=4e-5)

InputType = typing.TypeVar("InputType")


def train(
    model: "torch.nn.Module",
    loss: typing.Callable[[typing.List[InputType]], "torch.Tensor"],
    training: typing.List[InputType],
    validation: typing.List[InputType] = None,
    batch_size: int = 1,
    augment: typing.Callable[[typing.List[InputType]], typing.List[InputType]] = None,
    epochs=100,
    on_epoch_start: typing.Callable = None,
    on_epoch_end: typing.Callable[[typing.List[dict]], dict] = None,
    shuffle=True,
    optimizer_params=None,
    scheduler_params=None,
    clip_grad_norm_params=None,
):
    """Run training job.
    Args:
        model: The model that we're training.
        loss: A function to compute the loss for a batch.
        training: The collection of training images
        validation: The collection of validation images
        batch_size: The batch size to use for training
        augmenter: The augmenter for generating samples
        epochs: The number of epochs to train.
        on_epoch_start: A callback to run when starting a new epoch.
        on_epoch_end: A callback to run when finishing an epoch.
        shuffle: Whether to shuffle the training data on each epoch.
        optimizer_params: Passed to timm.optim.create_optimizer_v2 to build
            the optimizer.
        scheduler_params: Passed to timm.scheduler.create_scheduler to build
            the scheduler.
    """
    assert timm is not None, "timm is required for this function"
    assert torch is not None, "torch is required for this function."
    optimizer = timm.optim.create_optimizer_v2(
        model, **(optimizer_params or DEFAULT_OPTIMIZER_PARAMS)
    )
    scheduler, _ = timm.scheduler.create_scheduler(
        types.SimpleNamespace(**(scheduler_params or DEFAULT_SCHEDULER_PARAMS)),
        optimizer=optimizer,
    )
    train_index = np.arange(len(training)).tolist()
    summaries: typing.List[typing.Dict[str, typing.Any]] = []
    terminated = False
    try:
        for epoch in range(epochs):
            with tqdm.trange(len(training) // batch_size) as t:
                model.train()
                t.set_description(f"Epoch {epoch + 1} / {epochs}")
                scheduler.step(
                    epoch=epoch,
                    metric=None
                    if not summaries
                    else summaries[-1].get("val_loss", summaries[-1]["loss"]),
                )
                if on_epoch_start:
                    on_epoch_start()
                cum_loss = 0
                for batchIdx, start in enumerate(range(0, len(training), batch_size)):
                    if batchIdx == 0 and shuffle:
                        random.shuffle(train_index)
                    end = min(start + batch_size, len(train_index))
                    batch = [training[train_index[idx]] for idx in range(start, end)]
                    if augment:
                        batch = augment(batch)
                    optimizer.zero_grad()
                    batch_loss = loss(batch)
                    batch_loss.backward()
                    if clip_grad_norm_params is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), **clip_grad_norm_params
                        )
                    cum_loss += batch_loss.detach().cpu().numpy()
                    avg_loss = cum_loss / end
                    optimizer.step()
                    t.set_postfix(loss=avg_loss)
                    t.update()
                summaries.append({"loss": avg_loss})
                if validation:
                    summaries[-1]["val_loss"] = np.sum(
                        [
                            loss(
                                [
                                    validation[idx]
                                    for idx in range(
                                        vstart,
                                        min(vstart + batch_size, len(validation)),
                                    )
                                ]
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            for vstart in range(0, len(validation), batch_size)
                        ]
                    ) / len(validation)
                summaries[-1]["lr"] = next(g["lr"] for g in optimizer.param_groups)
                if on_epoch_end:
                    try:
                        summaries[-1] = {**summaries[-1], **on_epoch_end(summaries)}
                    except StopIteration:
                        terminated = True
                t.set_postfix(**summaries[-1])
            if terminated:
                break
    except KeyboardInterrupt:
        LOGGER.warning("Terminating early due to keyboard interrupt.")
        return summaries
    return summaries
