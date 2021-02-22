import os
import math
import logging
import argparse
import tensorflow as tf
from tensorflow.python.platform import tf_logging

import dbh_util as util
import classifier as cl
import pandas2tf

EPS = 1e-8
CLASS_NUM = 2
MAX_MISSES = 4

def log(msg):
    tf_logging.log(tf_logging.FATAL, msg) # FATAL to show up at any TF logging level
    logging.getLogger('DeepBugHunter').info(msg)

#
# Strategy args
#

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, help='Number of layers')
parser.add_argument('--neurons', type=int, help='Number of neurons per layer')
parser.add_argument('--batch', type=int, help='Batch size')
parser.add_argument('--epochs', type=int, help='Epoch count')
parser.add_argument('--lr', type=float, help='Starting learning rate')
parser.add_argument('--beta', type=float, default=0.0, help='L2 regularization bias')
parser.add_argument('--sandbox', default=os.path.abspath('sandbox'), help='Intermediary model folder')

#
# Simple DNN classification for a set number of epochs
#

def to_tf_dataset(src_data, batch_size):
    return tf.data.Dataset.from_tensor_slices((src_data[0].values, src_data[1].values)).batch(batch_size, drop_remainder=True)

def preds_to_classes(preds):
    return (preds[:, 0] >= 0.5).astype(int)

def predict(classifier, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())
    preds = classifier.predict(to_tf_dataset(test, sargs['batch']), batch_size = sargs['batch'])
    return preds_to_classes(preds)

def results_to_dict(results):
    keys = ['loss', 'accuracy', 'precision', 'recall', 'tp', 'tn', 'fp', 'fn', 'fmes']
    ret = dict(zip(keys, results))

    #TODO: Fix this if MCC is ready
    ret['mcc'] = -1
    
    return ret

def learn(train, dev, test, args, sargs_str):

    # Read strategy-specific args
    sargs = util.parse(parser, sargs_str.split())

    # Clean out the sandbox
    util.mkdir(sargs['sandbox'], clean=True)
   
    # Feature columns describe how to use the input
    my_feature_columns = []
    for key in train[0].keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Calculate epoch length
    steps_per_epoch = math.ceil(len(train[0]) / sargs['batch'])
    total_steps = sargs['epochs'] * steps_per_epoch

    # Train a classifier
    extra_args = {
        'class_num': CLASS_NUM,
        'columns': my_feature_columns,
        'steps_per_epoch': steps_per_epoch,
        'learning_rate': sargs['lr'],
        'model_dir': sargs['sandbox'],
        'warm_start_dir': None,
        'feature_num' : train[0].shape[1]
    }
    merged_args = {**args, **sargs, **extra_args}

    # Create a new classifier instance
    classifier = cl.create_classifier(merged_args)

    # Train the model for exactly 1 epoch
    train_set = to_tf_dataset(train, sargs['batch'])
    classifier.fit(
        train_set,
        epochs = sargs['epochs'],
        batch_size = sargs['batch'])

    # Evaluate the model
    train_result = classifier.evaluate(train_set, batch_size = sargs['batch'])
    dev_result = classifier.evaluate(to_tf_dataset(dev, sargs['batch']), batch_size = sargs['batch'])
    test_result = classifier.evaluate(to_tf_dataset(test, sargs['batch']), batch_size = sargs['batch'])

    train_result = results_to_dict(train_result)
    dev_result = results_to_dict(dev_result)
    test_result = results_to_dict(test_result)
    
    return train_result, dev_result, test_result, classifier