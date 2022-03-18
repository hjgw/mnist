import tensorflow as tf
from absl import app as absl_app
from absl import flags
import sys
from mnist import run_mnist
from utils.flags import core as flags_core
from utils.flags.core import help_wrap


def define_mnist_flags():
    key_flags = []
    flags.DEFINE_string(
        name="data_dir", short_name="dd", default="/dlspace/volumes/data/",
        help=help_wrap("The location of the input data."))
    key_flags.append("data_dir")

    flags.DEFINE_string(
        name="model_dir", short_name="md", default="/tmp",
        help=help_wrap("The location of the model checkpoint files."))
    key_flags.append("model_dir")

    flags.DEFINE_boolean(
        name="clean", default=False,
        help=help_wrap("If set, model_dir will be removed if it exists."))
    key_flags.append("clean")

    flags.DEFINE_integer(
        name="train_epochs", short_name="te", default=3,
        help=help_wrap("The number of epochs used to train."))
    key_flags.append("train_epochs")

    flags.DEFINE_integer(
        name="epochs_between_evals", short_name="ebe", default=1,
        help=help_wrap("The number of training epochs to run between "
                       "evaluations."))
    key_flags.append("epochs_between_evals")

    flags.DEFINE_float(
        name="stop_threshold", short_name="st",
        default=None,
        help=help_wrap("If passed, training will stop at the earlier of "
                       "train_epochs and when the evaluation metric is  "
                       "greater than or equal to stop_threshold."))

    flags.DEFINE_integer(
        name="batch_size", short_name="bs", default=32,
        help=help_wrap("Batch size for training and evaluation. When using "
                       "multiple gpus, this is the global batch size for "
                       "all devices. For example, if the batch size is 32 "
                       "and there are 4 GPUs, each GPU will get 8 examples on "
                       "each step."))
    key_flags.append("batch_size")

    flags.DEFINE_integer(
        name="num_gpus", short_name="ng",
        default=1 if tf.test.is_gpu_available() else 0,
        help=help_wrap(
            "How many GPUs to use with the DistributionStrategies API. The "
            "default is 1 if TensorFlow can detect a GPU, and 0 otherwise."))
    key_flags.append("num_gpus")

    flags.DEFINE_string(
        name="export_dir", short_name="ed", default=None,
        help=help_wrap("If set, a SavedModel serialization of the model will "
                       "be exported to this directory at the end of training. "
                       "See the README for more details and relevant links.")
    )
    key_flags.append("export_dir")
    flags_core.define_performance(num_parallel_calls=False)
    flags_core.define_image()
    return key_flags


def main(_):
    run_mnist(flags.FLAGS)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_mnist_flags()
    absl_app.run(main)
