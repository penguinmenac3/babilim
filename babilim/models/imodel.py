from typing import Any
from babilim import status, info, warn, error, DEBUG_VERBOSITY
from babilim.core import Tensor, RunOnlyOnce, GradientTape
from babilim.layers.ilayer import ILayer
from babilim.data import Dataset, Dataloader
from babilim.experiment import Config
from babilim.optimizers.learning_rates import LearningRateSchedule
from babilim.core.checkpoint import Checkpoint
import babilim.core.statefull_object as so
import os
import time
import numpy as np
from tensorboardX import SummaryWriter


class IModel(ILayer):
    def __init__(self, layer_type: str = "Model"):
        super().__init__(layer_type)
        self.uninitialized = True

    def __dict_to_str(self, data):
        out = []
        for k in data:
            if isinstance(data[k], list):
                for i in data[k]:
                    name = i.__name__
                    out.append("{}_{}={:.3f}".format(k, name, data[k]))
            else:
                out.append("{}={:.3f}".format(k, data[k]))
        return " - ".join(out)

    def __format_time(self, t):
        hours, remainder = divmod(t, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '%d:%02d:%02d' % (hours, minutes, seconds)

    def run_epoch(self, config: Config, dataset, optimizer, lr_schedule, loss, metrics, samples_seen: int, summary_writer):
        """
        Run an epoch in training or validation.

        (This function is called in fit and it is NOT RECOMMENDED to use this function from outside.)

        Optimizer is "optional" if it is set to None, it is a validation run otherwise it is a training run.

        :param config: The configuration for the run.
        :param dataset: The native dataset.
        :param optimizer: The babilim optimizer or None for validation.
        :param lr_schedule: The learning rate scheduler (also required for validation).
        :param loss: The loss function.
        :param metrics: The metric computation function.
        :param samples_seen: The number of samples the network has seen before running this method.
        :param summary_writer: The summary writer where to store the summaries.
        :return: Returns the average loss and metrics.
        """
        N = len(dataset)

        # Setup the training loop
        variables = self.trainable_variables

        # Loop over the dataset and update weights.
        for i, (x, y) in enumerate(dataset):
            # Forward pass, computing gradients and applying them
            with GradientTape(variables) as tape:
                prediction = self(**x._asdict())
                for p in prediction:
                    if p.is_nan().any():
                        error("NaN NetworkOutput {}: {}".format(p.name, p.native))
                        raise ValueError("NetworkOutput {} got nan.".format(p.name))
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
                    loss.summary(samples_seen, summary_writer)
                    metrics.summary(samples_seen, summary_writer)
                    loss.reset_avg()
                    metrics.reset_avg()
            else:
                status("Validating {}/{} - Loss {:.3f}".format(i + 1, N, loss_val), end="")
        if optimizer is None:
            loss.summary(samples_seen, summary_writer)
            metrics.summary(samples_seen, summary_writer)
        print()
        return loss.avg, metrics.avg

    def _init_model(self, batched_training_dataset, chkpt_path, config, optim, loss, metrics, lr_schedule, train_summary_writer):
        samples_seen = 0

        # Actually force model to be build by running one forward step
        if DEBUG_VERBOSITY:
            info("Build model.")
        if self.uninitialized:
            self.uninitialized = False
            features, _ = next(iter(batched_training_dataset))
            self(**features._asdict())
            # TODO implement model graph logging sooner or later.
            #train_summary_writer.add_graph(self.model, features)

        # Load Checkpoint
        epoch = 0
        saved_models_path = os.path.join(chkpt_path, "checkpoints")
        saved_models = sorted([os.path.join(saved_models_path, f) for f in os.listdir(saved_models_path)])
        if len(saved_models) > 0 and os.path.exists(saved_models[-1]):
            info("Loading checkpoint: {}".format(saved_models[-1]))
            checkpoint = Checkpoint(saved_models[-1])
            if DEBUG_VERBOSITY:
                checkpoint.print()
            epoch = checkpoint.get_epoch() + 1
            samples_seen = len(batched_training_dataset) * config.train_batch_size * epoch
            model_state = checkpoint.get_state_dict("model")
            optim_state = checkpoint.get_state_dict("optimizer")
            loss_state = checkpoint.get_state_dict("loss")
            metrics_state = checkpoint.get_state_dict("metrics")
            lr_schedule_state = checkpoint.get_state_dict("lr_schedule")
            if len(model_state) > 0:
                if DEBUG_VERBOSITY:
                    info("Load Model...")
                self.load_state_dict(model_state)
            else:
                warn("Could not find model_state in checkpoint.")
            if len(optim_state) > 0:
                if DEBUG_VERBOSITY:
                    info("Load Optimizer...")
                optim.load_state_dict(optim_state)
            else:
                warn("Could not find optimizer_state in checkpoint.")
            if len(loss_state) > 0:
                if DEBUG_VERBOSITY:
                    info("Load Loss...")
                loss.load_state_dict(loss_state)
            else:
                warn("Could not find loss_state in checkpoint.")
            if len(metrics_state) > 0:
                if DEBUG_VERBOSITY:
                    info("Load Metrics...")
                metrics.load_state_dict(metrics_state)
            else:
                warn("Could not find metrics_state in checkpoint.")
            if len(lr_schedule_state) > 0:
                if DEBUG_VERBOSITY:
                    info("Load LR Schedule...")
                lr_schedule.load_state_dict(lr_schedule_state)
            else:
                warn("Could not find lr_schedule_state in checkpoint.")

        if DEBUG_VERBOSITY:
            info("Trainable Variables:")
            for name, var in self.named_trainable_variables.items():
                info("  {}: {}".format(name, var.shape))
            info("Untrainable Variables:")
            for name, var in self.named_untrainable_variables.items():
                info("  {}: {}".format(name, var.shape))

        return epoch, samples_seen

    def fit(self, training_dataset: Dataset, validation_dataset: Dataset, loss, metrics, config: Config, optim: Any,
            lr_schedule: LearningRateSchedule, verbose: bool = True):
        config.check_completness()
        if config.train_actual_checkpoint_path is None:
            raise RuntimeError(
                "You must setup logger before calling the fit method. See babilim.experiment.logger.setup")
        chkpt_path = config.train_actual_checkpoint_path

        # Summary writers
        train_summary_writer = SummaryWriter(os.path.join(chkpt_path, "train"))
        val_summary_writer = SummaryWriter(os.path.join(chkpt_path, "val"))

        # Create batched dataloaders.
        training_dataloader = training_dataset.to_dataloader()
        validation_dataloader = validation_dataset.to_dataloader()

        # Try to retrieve optional arguments from hyperparams if not specified
        epochs = config.train_epochs

        epoch, samples_seen = self._init_model(training_dataloader, chkpt_path, config, optim, loss, metrics, lr_schedule, train_summary_writer)

        info("Start training for {} epochs from epoch {}.".format(epochs, epoch))
        start = time.time()
        for i in range(epoch, epochs):
            loss.reset_avg()
            metrics.reset_avg()
            self.train()
            self.run_epoch(config, training_dataloader, optim, lr_schedule, loss, metrics, samples_seen, train_summary_writer)
            samples_seen += len(training_dataloader) * config.train_batch_size

            # save checkpoint
            checkpoint = Checkpoint(os.path.join(chkpt_path, "checkpoints", "chkpt_{:09d}.npz".format(i)))
            checkpoint.set_epoch(i)
            checkpoint.set_state_dict("model", self.state_dict())
            checkpoint.set_state_dict("optimizer", optim.state_dict())
            checkpoint.set_state_dict("loss", loss.state_dict())
            checkpoint.set_state_dict("metrics", metrics.state_dict())
            checkpoint.set_state_dict("lr_schedule", lr_schedule.state_dict())
            checkpoint.save()

            loss.reset_avg()
            metrics.reset_avg()
            self.eval()
            loss_results, metrics_results = self.run_epoch(config, validation_dataloader, None, lr_schedule, loss, metrics, samples_seen, val_summary_writer)
            elapsed_time = time.time() - start
            eta = elapsed_time / (i + 1) * (epochs - (i + 1))
            status("Epoch {}/{} - ETA {} - {} - {}".format(i + 1, epochs, self.__format_time(eta),
                                                           self.__dict_to_str(loss_results),
                                                           self.__dict_to_str(metrics_results)))

            # Load checkpoint again.
            #epoch, samples_seen = self._init_model(training_dataloader, chkpt_path, config, optim, loss, metrics, lr_schedule, verbose)

    def load(self, model_file):
        checkpoint = Checkpoint(model_file)
        if DEBUG_VERBOSITY:
            checkpoint.print()
        model_state = checkpoint.get_state_dict("model")
        if len(model_state) > 0:
            self.load_state_dict(model_state)
        else:
            error("Could not find model_state_dict in checkpoint.")

    def save(self, model_file):
        checkpoint = Checkpoint(model_file)
        checkpoint.set_state_dict("model", self.state_dict())
        if DEBUG_VERBOSITY:
            checkpoint.print()
        checkpoint.save()

    def predict(self, **kwargs):
        """
        Pass in single training examples as numpy arrays.

        The array must not have batch dimension.

        :param kwargs: The parameters to feed the network as a single example.
        :return: The output for a single example.
        """
        kwargs = {k: np.array([kwargs[k]]) for k in kwargs.keys() if isinstance(kwargs[k], np.ndarray)}
        kwargs = {k: Tensor(data=kwargs[k], trainable=False) for k in kwargs.keys()}

        preds = self.__call__(**kwargs)
        tmp = preds._asdict()
        tmp = {k: tmp[k].numpy()[0] for k in tmp.keys()}
        preds = type(preds)(**tmp)
        return preds

    def eval(self):
        so.TRAINING = False

    def train(self):
        so.TRAINING = True


class NativeModelWrapper(IModel):
    def __init__(self, model, layer_type: str = "NativeModel"):
        """
        Wrap a native model to be trained and used as a babilim model.

        The model can have a build function which is then used, but it does not need to.

        :param model: The model that should be wrapped.
        :param layer_type: The layer type. Defaults to NativeModel.
        """
        super().__init__(layer_type=layer_type)
        self.native_model = model

    @RunOnlyOnce
    def build(self, *args, **kwargs) -> None:
        build = getattr(self.native_model, "build", None)
        if callable(build):
            # Unwrap arguments
            args = [feature.native for feature in args]
            kwargs = {k: kwargs[k].native for k in kwargs}

            # Call the build
            build(*args, **kwargs)

    def call(self, *args, **kwargs) -> Any:
        # Unwrap arguments
        args = [feature.native for feature in args]
        kwargs = {k: kwargs[k].native for k in kwargs}

        # call function
        result = self.native_model(*args, **kwargs)
        result_raw = result._asdict()

        # Wrap results
        result_raw = {k: Tensor(data=result_raw[k], trainable=True) for k in result_raw}
        return type(result)(**result_raw)

    def eval(self):
        so.TRAINING = False
        self.native_model.eval()

    def train(self):
        so.TRAINING = True
        self.native_model.train()
