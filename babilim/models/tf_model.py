from typing import Sequence, Any, Sequence, Callable, Dict
from babilim.experiment.logging import tprint
from typing import Sequence
import datetime, time
import os
import time
import tensorflow as tf

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
                if isinstance(i, tf.Module):
                    name = i.name
                out.append("{}_{}={:.3f}".format(k, name, data[k].numpy()))
        else:
            out.append("{}={:.3f}".format(k, data[k].numpy()))
    return " - ".join(out)


def format_time(t):
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)


#@tf.function
def _train(config: Config, model, dataset: Sequence, optimizer, lr_schedule, loss, metrics, samples_seen: int, verbose: bool):
    N = len(dataset)
    
    # Setup the training loop
    loss.reset_avg()
    metrics.reset_avg()

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
        gradients = tape.gradient(loss_results)
        lr = lr_schedule(samples_seen / config.train_batch_size)
        optimizer.apply_gradients(gradients, variables, lr)
        
        # Update global variables and log the variables
        samples_seen += config.train_batch_size
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H.%M.%S')
        tprint("Training {}/{} - Loss {:.3f} - LR {:.6f}".format(i + 1, N, loss.avg["total"].numpy(), lr), end="")
        if i % config.train_log_steps == 0:
            tf.summary.scalar('learning_rate', lr, step=samples_seen)
            loss.summary(samples_seen)
            metrics.summary(samples_seen)
    print()


#@tf.function
def _validate(config, model, dataset: Sequence, loss, metrics, samples_seen):
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
    loss.summary(samples_seen)
    metrics.summary(samples_seen)
    print()
    return loss.avg, metrics.avg


def fit(model, training_dataset: Dataset, validation_dataset: Dataset, loss, metrics, config: Config, optim: Any, lr_schedule: LearningRateSchedule, verbose: bool):
    config.check_completness()
    if config.train_actual_checkpoint_path is None:
        raise RuntimeError("You must setup logging before calling the fit method. See babilim.experiment.logging.setup")
    chkpt_path = config.train_actual_checkpoint_path

    # Summary writers
    train_summary_writer = tf.summary.create_file_writer(os.path.join(chkpt_path, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(chkpt_path, "val"))

    # Try to retrieve optional arguments from hyperparams if not specified
    epochs = config.train_epochs
    
    batched_training_dataset = training_dataset.to_keras()
    batched_validation_dataset = validation_dataset.to_keras()

    # Load Checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1))
    manager = tf.train.CheckpointManager(ckpt, os.path.join(chkpt_path, "checkpoints"), max_to_keep=10)
    ckpt.restore(manager.latest_checkpoint)

    # Actually force model to be build by running one forward step
    tprint("Build model.")
    features, _ = batched_training_dataset[0]
    inp, _ = _tensor_wrapper.wrap(features._asdict())
    model(**inp)

    variables = model.trainable_variables
    if verbose:
        print()
        print("*****************************")
        print("* model.trainable_variables *")
        print("*****************************")
        for var in variables:
            print("  {}: {}".format(var.name, var.shape))
        print()

    tprint("Start training for {} epochs.".format(epochs))
    samples_seen = 0
    start = time.time()
    for i in range(epochs):
        loss.reset_avg()
        metrics.reset_avg()
        with train_summary_writer.as_default():
            _train(config, model, batched_training_dataset, optim, lr_schedule, loss, metrics, samples_seen, verbose)
            samples_seen += len(batched_training_dataset) * config.train_batch_size
        
        loss.reset_avg()
        metrics.reset_avg()
        with val_summary_writer.as_default():
            loss_results, metrics_results = _validate(config, model, batched_validation_dataset, loss, metrics, samples_seen)

        ckpt.step.assign_add(1)
        save_path = manager.save()
        elapsed_time = time.time() - start
        eta = elapsed_time / (i + 1) * (epochs - (i + 1))
        tprint("Epoch {}/{} - ETA {} - {} - {}".format(i + 1, epochs, format_time(eta),
                                                        __dict_to_str(loss_results), __dict_to_str(metrics_results)))
