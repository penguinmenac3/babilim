from typing import Sequence

import os
import time
import tensorflow as tf
from babilim.data import Dataset
from babilim.experiment import Config


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
def _train(config: Config, model, dataset: Sequence, optimizer, loss, metrics, samples_seen: int):
    N = len(dataset)
    
    # Setup the training loop
    loss.reset_avg()
    metrics.reset_avg()

    # Loop over the dataset and update weights.
    for i, (x, y) in enumerate(dataset):
        # Forward pass, computing gradients and applying them
        with tf.GradientTape() as tape:
            prediction = model(**x)
            loss_results = loss(y, prediction)
            metrics(y, prediction)
        variables = model.trainable_variables
        gradients = tape.gradient(loss_results, variables)
        # FIXME update optimizer.lr
        optimizer.apply_gradients(zip(gradients, variables))
        
        # Update global variables and log the variables
        samples_seen += config.train_batch_size
        # FIXME TypeError: unsupported format string passed to Tensor.__format__
        print("\rTraining {}/{} - Loss {:.3f}".format(i + 1, N, loss.avg["total"]), end="")
        if i % config.train_log_steps == 0:
            tf.summary.scalar('learning_rate', optimizer.lr, step=samples_seen)
            loss.summary()
            metrics.summary()
    print()


#@tf.function
def _validate(config, model, dataset: Sequence, loss, metrics):
    N = len(dataset)
    for i, (x, y) in enumerate(dataset):
        prediction = model(**x)
        loss(y, prediction)
        metrics(y, prediction)
        print("\rValidating {}/{} - Loss {:.3f}".format(i, N, loss.avg["total"]), end="")
    loss.summary()
    metrics.summary()
    print()
    return loss.avg, metrics.avg


def fit(model, training_dataset: Dataset, validation_dataset: Dataset, config: Config):
    config.check_completness()
    if config.train_actual_checkpoint_path is None:
        raise RuntimeError("You must setup logging before calling the fit method. See babilim.experiment.logging.setup")
    chkpt_path = config.train_actual_checkpoint_path

    # Summary writers
    train_summary_writer = tf.summary.create_file_writer(os.path.join(chkpt_path, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(chkpt_path, "val"))

    # Try to retrieve optional arguments from hyperparams if not specified
    loss = config.arch_loss(config)
    metrics = config.arch_metrics(config)
    optimizer = config.train_optimizer(config)
    lr_scheduler = config.train_learning_rate_shedule(config)
    epochs = config.train_epochs
    model.optimizer = optimizer
    lr_scheduler.model = model
    
    batched_training_dataset = training_dataset.to_keras()
    batched_validation_dataset = validation_dataset.to_keras()

    # Load Checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(chkpt_path, "checkpoints"), max_to_keep=10)
    ckpt.restore(manager.latest_checkpoint)

    print("Epoch {}/{}".format(1, epochs))
    samples_seen = 0
    start = time.time()
    for i in range(epochs):
        loss.reset_avg()
        metrics.reset_avg()
        with train_summary_writer.as_default():
            _train(config, model, batched_training_dataset, optimizer, loss, metrics, samples_seen)
            samples_seen += len(batched_training_dataset)
        
        loss.reset_avg()
        metrics.reset_avg()
        with val_summary_writer.as_default():
            loss_results, metrics_results = _validate(config, model, batched_validation_dataset, loss, metrics)

        ckpt.step.assign_add(1)
        save_path = manager.save()
        elapsed_time = time.time() - start
        eta = elapsed_time / (i + 1) * (epochs - (i + 1))
        print("\rEpoch {}/{} - ETA {} - {} - {}".format(i + 1, epochs, format_time(eta),
                                                        __dict_to_str(loss_results), __dict_to_str(metrics_results)))
