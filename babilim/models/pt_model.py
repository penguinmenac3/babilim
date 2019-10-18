from typing import Sequence, Any, Sequence, Callable, Dict
from babilim.experiment.logging import tprint
from typing import Sequence
import datetime, time
import os
import time

from tensorboardX import SummaryWriter
import torch

from babilim.core import GradientTape
from babilim.data import Dataset
from babilim.experiment import Config
from babilim.core.tensor import TensorWrapper
from babilim.optimizers.learning_rates import LearningRateSchedule


_tensor_wrapper = TensorWrapper()


def __dict_to_str(data):
    out = []
    for k in data:
        if isinstance(data[k], list):
            for i in data[k]:
                name = i.__name__
                out.append("{}_{}={:.3f}".format(k, name, data[k].numpy()))
        else:
            out.append("{}={:.3f}".format(k, data[k].numpy()))
    return " - ".join(out)


def format_time(t):
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)


def _train(config: Config, model, dataset, optimizer, lr_schedule, loss, metrics, samples_seen: int, summary_writer, verbose: bool):
    N = len(dataset)

    # Setup the training loop
    loss.reset_avg()
    metrics.reset_avg()
    
    # Setup the training loop
    variables = model.trainable_variables

    # Loop over the dataset and update weights.
    for i, (x, y) in enumerate(dataset):
        # Forward pass, computing gradients and applying them
        with GradientTape(variables) as tape:
            inp, _ = _tensor_wrapper.wrap(x._asdict())
            outp, _ = _tensor_wrapper.wrap(y._asdict())
            outp = type(y)(**outp)
            prediction = model(**inp)
            loss_results = loss(y_true=outp, y_pred=prediction)
            loss.log("total", loss_results)
            metrics(y_true=outp, y_pred=prediction)
        # Translate those to something usefull...
        gradients = tape.gradient(loss_results)
        lr = lr_schedule(samples_seen / config.train_batch_size)
        optimizer.apply_gradients(gradients, variables, lr)
        
        # Update global variables and log the variables
        samples_seen += config.train_batch_size
        tprint("Training {}/{} - Loss {:.3f} - LR {:.6f}".format(i + 1, N, loss.avg["total"].numpy(), lr), end="")
        if i % config.train_log_steps == 0:
            summary_writer.add_scalar('learning_rate', lr, global_step=samples_seen)
            loss.summary(samples_seen, summary_writer)
            metrics.summary(samples_seen, summary_writer)
    print()


def _validate(config, model, dataset, loss, metrics, samples_seen, summary_writer):
    N = len(dataset)
    for i, (x, y) in enumerate(dataset):
        inp, _ = _tensor_wrapper.wrap(x._asdict())
        outp, _ = _tensor_wrapper.wrap(y._asdict())
        outp = type(y)(**outp)
        prediction = model(**inp)
        loss_results = loss(y_true=outp, y_pred=prediction)
        loss.log("total", loss_results)
        metrics(y_true=outp, y_pred=prediction)
        tprint("Validating {}/{} - Loss {:.3f}".format(i, N, loss.avg["total"].numpy()), end="")
    loss.summary(samples_seen, summary_writer)
    metrics.summary(samples_seen, summary_writer)
    print()
    return loss.avg, metrics.avg


def fit(model, training_dataset: Dataset, validation_dataset: Dataset, loss, metrics, config: Config, optim: Any, lr_schedule: LearningRateSchedule, verbose: bool):
    config.check_completness()
    if config.train_actual_checkpoint_path is None:
        raise RuntimeError("You must setup logging before calling the fit method. See babilim.experiment.logging.setup")
    chkpt_path = config.train_actual_checkpoint_path

    # Summary writers
    train_summary_writer = SummaryWriter(os.path.join(chkpt_path, "train"))
    val_summary_writer = SummaryWriter(os.path.join(chkpt_path, "val"))

    # Try to retrieve optional arguments from hyperparams if not specified
    epochs = config.train_epochs
    
    batched_training_dataset = training_dataset.to_pytorch()
    batched_validation_dataset = validation_dataset.to_pytorch()

    # Actually force model to be build by running one forward step
    tprint("Build model.")
    features, _ = next(iter(batched_training_dataset))
    inp, _ = _tensor_wrapper.wrap(features._asdict())
    model(**inp)
    
    # Load Checkpoint
    epoch = 0
    saved_models_path = os.path.join(chkpt_path, "checkpoints")
    saved_models = sorted([os.path.join(saved_models_path, f) for f in os.listdir(saved_models_path)])
    if len(saved_models) > 0 and os.path.exists(saved_models[-1]):
        tprint("Loading checkpoint: {}".format(saved_models[-1]))
        checkpoint = torch.load(saved_models[-1])
        epoch = checkpoint["epoch"] + 1
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            tprint("WARNING: Could not find model_state_dict in checkpoint.")
        if "optimizer_state_dict" in checkpoint:
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            tprint("WARNING: Could not find optimizer_state_dict in checkpoint.")
        if "loss_state_dict" in checkpoint:
            loss.load_state_dict(checkpoint['loss_state_dict'])
        else:
            tprint("WARNING: Could not find loss_state_dict in checkpoint.")
        if "metrics_state_dict" in checkpoint:
            metrics.load_state_dict(checkpoint['metrics_state_dict'])
        else:
            tprint("WARNING: Could not find metrics_state_dict in checkpoint.")
        if "lr_schedule_state_dict" in checkpoint:
            lr_schedule.load_state_dict(checkpoint['lr_schedule_state_dict'])
        else:
            tprint("WARNING: Could not find lr_schedule_state_dict in checkpoint.")

    variables = model.trainable_variables
    if verbose:
        print()
        print("*****************************")
        print("* model.trainable_variables *")
        print("*****************************")
        for var in variables:
            print("  {}: {}".format(var.name, var.shape))
        print()

    tprint("Start training for {} epochs from epoch {}.".format(epochs, epoch))
    samples_seen = len(batched_training_dataset) * config.train_batch_size * epoch
    start = time.time()
    for i in range(epoch, epochs):
        loss.reset_avg()
        metrics.reset_avg()
        _train(config, model, batched_training_dataset, optim, lr_schedule, loss, metrics, samples_seen, train_summary_writer, verbose)
        samples_seen += len(batched_training_dataset) * config.train_batch_size
        
        loss.reset_avg()
        metrics.reset_avg()
        loss_results, metrics_results = _validate(config, model, batched_validation_dataset, loss, metrics, samples_seen, val_summary_writer)

        # save checkpoint
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss_state_dict': loss.state_dict(),
            'metrics_state_dict': metrics.state_dict(),
            'lr_schedule_state_dict': lr_schedule.state_dict(),
        }, os.path.join(chkpt_path, "checkpoints", "chkpt_{:09d}.pt".format(i)))

        elapsed_time = time.time() - start
        eta = elapsed_time / (i + 1) * (epochs - (i + 1))
        tprint("Epoch {}/{} - ETA {} - {} - {}".format(i + 1, epochs, format_time(eta),
                                                        __dict_to_str(loss_results), __dict_to_str(metrics_results)))
