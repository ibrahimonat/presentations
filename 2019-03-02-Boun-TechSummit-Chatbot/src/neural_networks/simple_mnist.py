import argparse
import numpy as np
import sys
import os
import cntk as C
from cntk.train import Trainer, minibatch_size_schedule 
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.device import cpu, try_set_default_device
from cntk.learners import adagrad, learning_parameter_schedule_per_sample
from cntk.ops import relu, element_times, constant
from cntk.layers import Dense, Sequential, For
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.train.training_session import *
from cntk.logging import ProgressPrinter, TensorBoardProgressWriter
import datetime

abs_path = os.path.dirname(os.path.abspath(__file__))

def check_path(path):
    if not os.path.exists(path):
        readme_file = os.path.normpath(os.path.join(
            os.path.dirname(path), "..", "README.md"))
        raise RuntimeError(
            "File '%s' does not exist. Please follow the instructions at %s to download and prepare it." % (path, readme_file))

def create_reader(path, is_training, input_dim, label_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features  = StreamDef(field='features', shape=input_dim, is_sparse=False),
        labels    = StreamDef(field='labels',   shape=label_dim, is_sparse=False)
    )), randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)


def create_learner(model):
    '''Create the optimized method'''
    optim = "momentum_sgd"
    lr = 0.001
    lr_per_sample = C.learning_parameter_schedule_per_sample(lr)
    momentum_schedule = C.momentum_schedule_per_sample(0.9990913221888589)
    if optim == 'momentum_sgd':
        clipping_threshold_per_sample = 5.0
        gradient_clipping_with_truncation = True
        return C.momentum_sgd(model.parameters, lr_per_sample, momentum_schedule,
                              gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                              gradient_clipping_with_truncation=gradient_clipping_with_truncation)

# Creates and trains a feedforward classification model for MNIST images

def simple_mnist(tensorboard_logdir=None):
    input_dim = 19
    num_output_classes = 2
    num_hidden_layers = 2
    hidden_layers_dim = 1024

    # Input variables denoting the features and label data
    feature = C.input_variable(input_dim, np.float32)
    label = C.input_variable(num_output_classes, np.float32)

    # Instantiate the feedforward classification model
    # scaled_input = element_times(constant(0.00390625), feature)

    z = Sequential([For(range(num_hidden_layers), lambda i: Dense(hidden_layers_dim, activation=relu)),
                    Dense(num_output_classes)])(feature)

    ce = cross_entropy_with_softmax(z, label)
    pe = classification_error(z, label)

    data_dir = r"."

    path = os.path.normpath(os.path.join(data_dir, "train.ctf"))
    check_path(path)

    reader_train = create_reader(path, True, input_dim, num_output_classes)

    input_map = {
        feature  : reader_train.streams.features,
        label  : reader_train.streams.labels
    }

    # Training config
    minibatch_size = 512
    num_samples_per_sweep = 1825000
    num_sweeps_to_train_with = 100

    # Instantiate progress writers.
    progress_writers = [ProgressPrinter(
        tag='Training',
        num_epochs=num_sweeps_to_train_with)]

    if tensorboard_logdir is not None:
        tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=tensorboard_logdir, model=z)
        progress_writers.append(tensorboard_writer)

    # Instantiate the trainer object to drive the model training
    lr = learning_parameter_schedule_per_sample(0.001)
    learner = create_learner(model=z)
    trainer = Trainer(z, (ce, pe), learner, progress_writers)

    num_minibatches_to_train = int(num_samples_per_sweep / minibatch_size * num_sweeps_to_train_with)

    model_dir = "model"
    for i in range(num_minibatches_to_train):
        mb = reader_train.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(mb)

        freq = int(num_samples_per_sweep / minibatch_size)
        if i > 0 and i % freq == 0:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            current_trainer_cp = os.path.join(model_dir, timestamp + "_epoch_" + str(freq) + ".trainer")
            trainer.save_checkpoint(current_trainer_cp)

            train_error = get_error_rate(os.path.join(data_dir, "train_subset.ctf"), input_map, input_dim,
                                         num_output_classes, trainer)
            valid_error = get_error_rate(os.path.join(data_dir, "validation.ctf"), input_map, input_dim, num_output_classes,
                                         trainer)

            if train_error > 0:
                tensorboard_writer.write_value("train_error", train_error, i)
            if valid_error > 0:
                tensorboard_writer.write_value("valid_error", valid_error, i)

    feat_path = os.path.normpath(os.path.join(data_dir, "test.ctf"))
    return get_error_rate(feat_path, input_map, input_dim, num_output_classes, trainer)


def get_error_rate(feat_path, input_map, input_dim, num_output_classes, trainer):
    check_path(feat_path)
    reader_test = create_reader(feat_path, False, input_dim, num_output_classes)

    test_minibatch_size = 1024
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    test_result = 0.0
    for i in range(0, int(num_minibatches_to_test)):
        mb = reader_test.next_minibatch(test_minibatch_size, input_map=input_map)
        eval_error = trainer.test_minibatch(mb)
        test_result = test_result + eval_error
    return test_result / num_minibatches_to_test


if __name__=='__main__':
    # try_set_default_device(cpu())

    parser = argparse.ArgumentParser()
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir',
                        help='Directory where TensorBoard logs should be created', required=False, default=None)
    args = vars(parser.parse_args())

    error = simple_mnist(args['tensorboard_logdir'])
    print("Error: %f" % error)