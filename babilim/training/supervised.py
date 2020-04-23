from typing import Any
import babilim
from babilim import status, info, warn, error
from babilim.core import Tensor, RunOnlyOnce, GradientTape
from babilim.core.logger import log_progress
from babilim.model.module import Module
from babilim.data import Dataset, Dataloader
from babilim.core import Config
from babilim.model.modules import Lambda
from babilim.training.optimizers.learning_rates import LearningRateSchedule
from babilim.core.checkpoint import Checkpoint
import os
import time
import numpy as np
from tensorboardX import SummaryWriter


def _dict_to_str(data):
    out = []
    for k in data:
        if isinstance(data[k], list):
            for i in data[k]:
                name = i.__name__
                out.append("{}_{}={:.3f}".format(k, name, data[k]))
        else:
            out.append("{}={:.3f}".format(k, data[k]))
    return " - ".join(out)


def _format_time(t):
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)


def run_epoch(model, config: Config, dataset, optimizer, lr_schedule, loss, metrics, samples_seen: int, summary_writer, goal=None):
    """
    Run an epoch in training or validation.

    (This function is called in fit and it is NOT RECOMMENDED to use this function from outside.)

    Optimizer is "optional" if it is set to None, it is a validation run otherwise it is a training run.

    :param config: The configuration for the run.
    :param dataset: The native dataset_class.
    :param optimizer: The babilim optimizer or None for validation.
    :param lr_schedule: The learning rate scheduler (also required for validation).
    :param loss: The loss function.
    :param metrics: The metric computation function.
    :param samples_seen: The number of detection the network has seen before running this method.
    :param summary_writer: The summary writer where to store the summaries.
    :return: Returns the average loss and metrics.
    """
    # wrap module if it was native
    if not isinstance(model, Module):
        model = Lambda(model)

    N = len(dataset)

    # Setup the training loop
    variables = model.trainable_variables

    # Set progress to zero.
    if goal is not None:
        log_progress(goal=goal, progress=0, score=0)
    loss_val = None

    # Loop over the dataset_class and update weights.
    for i, (x, y) in enumerate(dataset):
        # Forward pass, computing gradients and applying them
        with GradientTape(variables) as tape:
            prediction = model(**x._asdict())
            for name, p in prediction._asdict().items():
                if p.is_nan().any():
                    error("NaN NetworkOutput {}: {}".format(name, p.native))
                    raise ValueError("NetworkOutput {} got nan.".format(name))
            loss_results = loss(y_true=y, y_pred=prediction)
            loss.log("loss/total", loss_results)
            metrics(y_true=y, y_pred=prediction)

        loss_val = loss.avg["loss/total"]
        gradients = tape.gradient(loss_results)
        for grad in gradients:
            if grad is None:
                warn("A trainable variable did not have gradients."
                       "Did you set trainable or requires grads to false during your forward pass?")
                continue
            if grad.is_nan().any():
                error("NaN in gradient for {}: {}".format(grad.name, grad.native))
                raise ValueError("Gradient of {} got nan.".format(grad.name))
        lr = lr_schedule(samples_seen / config.train_batch_size)

        if optimizer is not None:
            # Translate those to something usefull...
            optimizer.apply_gradients(gradients, variables, lr)

            # Update global variables and log the variables
            samples_seen += config.train_batch_size
            status("Training {}/{} - Loss {:.3f} - LR {:.6f}".format(i + 1, N, loss_val, lr), end="")
            if i % config.train_log_steps == config.train_log_steps - 1:
                summary_writer.add_scalar('learning_rate', lr, global_step=samples_seen)
                if goal is not None:
                    log_progress(goal=goal, progress=((i + 1) / N), score=loss_val)
                loss.summary(samples_seen, summary_writer)
                metrics.summary(samples_seen, summary_writer)
                loss.reset_avg()
                metrics.reset_avg()
        else:
            status("Validating {}/{} - Loss {:.3f}".format(i + 1, N, loss_val), end="")
            if i % config.train_log_steps == config.train_log_steps - 1:
                if goal is not None:
                    log_progress(goal=goal, progress=((i + 1) / N), score=loss_val)

    if optimizer is None:
        loss.summary(samples_seen, summary_writer)
        metrics.summary(samples_seen, summary_writer)
    if goal is not None and loss_val is not None:
        log_progress(goal=goal, progress=1.0, score=loss_val)
    print()
    return loss.avg, metrics.avg


def _init_model(model: Module, batched_training_dataset, chkpt_path, config, optim, loss, metrics, lr_schedule, train_summary_writer, chkpt_native_format=False):
    samples_seen = 0

    # Actually force model to be build by running one forward step
    if not getattr(model, "initialized_model", False):
        if babilim.DEBUG_VERBOSITY:
            info("Build Model")
        model.initialized_model = True
        features, _ = next(iter(batched_training_dataset))
        model(**features._asdict())
        if isinstance(model, Lambda):
            # FIXME this does not work, pytorch needs dict support or my models need ordered parameters
            #print("Writing Graph to Tensorboard.")
            #train_summary_writer.add_graph(model.native_module, features)
            pass
        else:
            # TODO implement model graph logging sooner or later.
            pass

    # Load Checkpoint
    epoch = 0
    saved_models_path = os.path.join(chkpt_path, "checkpoints")
    saved_models = sorted([os.path.join(saved_models_path, f) for f in os.listdir(saved_models_path)])
    if len(saved_models) > 0 and os.path.exists(saved_models[-1]):
        info("Loading checkpoint: {}".format(saved_models[-1]))
        checkpoint = Checkpoint(saved_models[-1], chkpt_native_format)
        if babilim.DEBUG_VERBOSITY:
            checkpoint.print()
        epoch = checkpoint.get_epoch() + 1
        samples_seen = len(batched_training_dataset) * config.train_batch_size * epoch
        model_state = checkpoint.get_state_dict("model")
        optim_state = checkpoint.get_state_dict("optimizer")
        loss_state = checkpoint.get_state_dict("loss")
        metrics_state = checkpoint.get_state_dict("metrics")
        lr_schedule_state = checkpoint.get_state_dict("lr_schedule")
        if len(model_state) > 0:
            if babilim.DEBUG_VERBOSITY:
                info("Load Model...")
            model.load_state_dict(model_state)
        else:
            warn("Could not find model_state in checkpoint.")
        if len(optim_state) > 0:
            if babilim.DEBUG_VERBOSITY:
                info("Load Optimizer...")
            optim.load_state_dict(optim_state)
        else:
            warn("Could not find optimizer_state in checkpoint.")
        if len(loss_state) > 0:
            if babilim.DEBUG_VERBOSITY:
                info("Load Loss...")
            loss.load_state_dict(loss_state)
        else:
            warn("Could not find loss_state in checkpoint.")
        if len(metrics_state) > 0:
            if babilim.DEBUG_VERBOSITY:
                info("Load Metrics...")
            metrics.load_state_dict(metrics_state)
        else:
            warn("Could not find metrics_state in checkpoint.")
        if len(lr_schedule_state) > 0:
            if babilim.DEBUG_VERBOSITY:
                info("Load LR Schedule...")
            lr_schedule.load_state_dict(lr_schedule_state)
        else:
            warn("Could not find lr_schedule_state in checkpoint.")

    if babilim.DEBUG_VERBOSITY:
        info("Trainable Variables:")
        for name, var in model.named_trainable_variables.items():
            info("  {}: {}".format(name, var.shape))
        info("Untrainable Variables:")
        for name, var in model.named_untrainable_variables.items():
            info("  {}: {}".format(name, var.shape))

    return epoch, samples_seen


def fit(model, training_dataset: Dataset, validation_dataset: Dataset, loss, metrics, config: Config, optim: Any,
        lr_schedule: LearningRateSchedule, verbose: bool = True):
    try:
        # Wrap module if it is a native module.
        if not isinstance(model, Module):
            model = Lambda(model)

        config.check_completness()
        if config.train_actual_checkpoint_path is None:
            raise RuntimeError("You must setup logger before calling the fit method. See babilim.experiment.logger.setup")
        log_progress(goal="warmup", progress=0, score=0)
        chkpt_path = config.train_actual_checkpoint_path

        # Summary writers
        train_summary_writer = SummaryWriter(os.path.join(chkpt_path, "train"))
        val_summary_writer = SummaryWriter(os.path.join(chkpt_path, "val"))

        # Create batched dataloaders.
        training_dataloader = training_dataset.to_dataloader()
        validation_dataloader = validation_dataset.to_dataloader()

        # Try to retrieve optional arguments from hyperparams if not specified
        epochs = config.train_epochs

        chkpt_native_format = config.arch_chkpt_native_format

        epoch, samples_seen = _init_model(model, training_dataloader, chkpt_path, config, optim, loss, metrics, lr_schedule, train_summary_writer, chkpt_native_format)

        info("Start training for {} epochs from epoch {}.".format(epochs, epoch))
        start = time.time()
        log_progress(goal="train {}/{}".format(epoch + 1, epochs), progress=0, score=0)
        for i in range(epoch, epochs):
            loss.reset_avg()
            metrics.reset_avg()
            model.train()
            run_epoch(model, config, training_dataloader, optim, lr_schedule, loss, metrics, samples_seen, train_summary_writer, goal="train {}/{}".format(i + 1, epochs))
            samples_seen += len(training_dataloader) * config.train_batch_size

            # save checkpoint
            checkpoint = Checkpoint(os.path.join(chkpt_path, "checkpoints", "chkpt_{:09d}.npz".format(i)), native_format=chkpt_native_format)
            checkpoint.set_epoch(i)
            checkpoint.set_state_dict("model", model.state_dict())
            checkpoint.set_state_dict("optimizer", optim.state_dict())
            checkpoint.set_state_dict("loss", loss.state_dict())
            checkpoint.set_state_dict("metrics", metrics.state_dict())
            checkpoint.set_state_dict("lr_schedule", lr_schedule.state_dict())
            checkpoint.save()

            loss.reset_avg()
            metrics.reset_avg()
            model.eval()
            loss_results, metrics_results = run_epoch(model, config, validation_dataloader, None, lr_schedule, loss, metrics, samples_seen, val_summary_writer, goal="val {}/{}".format(i + 1, epochs))
            elapsed_time = time.time() - start
            eta = elapsed_time / (i + 1) * (epochs - (i + 1))
            status("Epoch {}/{} - ETA {} - {} - {}".format(i + 1, epochs, _format_time(eta),
                                                        _dict_to_str(loss_results),
                                                        _dict_to_str(metrics_results)))
        log_progress(goal="done", progress=1, score=loss.avg["loss/total"])
        if verbose:
            print("Training done.")
    except KeyboardInterrupt as e:
        log_progress(goal="paused", progress=1, score=loss.avg["loss/total"])
        print()
        print("Training stopped by user!")
    except Exception as e:
        log_progress(goal="failed", progress=1, score=loss.avg["loss/total"])
        raise e
