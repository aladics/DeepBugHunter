import os
import shutil
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
CLASSES = 2

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
parser.add_argument('--lr', type=float, help='Starting learning rate')
parser.add_argument('--beta', type=float, default=0.0, help='L2 regularization bias')
parser.add_argument('--max-misses', type=int, default=4, help='Maximum consecutive misses before early stopping')
parser.add_argument('--sandbox', default=os.path.abspath('sandbox'), help='Intermediary model folder')



#
# Validate after every epoch, and if the model gets worse, then restore the previous best model and try again
# with a reduced (halved) learning rate
#


def to_tf_dataset(src_data, batch_size):
    return tf.data.Dataset.from_tensor_slices((src_data[0].values, src_data[1].values)).batch(batch_size, drop_remainder=True)

def preds_to_classes(preds):
    return (preds[:, 0] >= 0.5).astype(int)

def results_to_dict(results):
    keys = ['loss', 'accuracy', 'precision', 'recall', 'tp', 'tn', 'fp', 'fn', 'fmes']
    ret = dict(zip(keys, results))

    #TODO: Fix this if MCC is ready
    ret['mcc'] = -1
    
    return ret

def predict(classifier, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())
    preds = classifier.predict(to_tf_dataset(test, sargs['batch']), batch_size = sargs['batch'])
    return preds_to_classes(preds)
    
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

    # Train a classifier
    # Repeat until the model consecutively "misses" a set number of times
    rounds = 1
    misses = miss_streak = 0
    best_result = {'fmes': -1}
    best_model_dir = None
    best_classifier = None

    train_set = to_tf_dataset(train, sargs['batch'])
    dev_set = to_tf_dataset(dev, sargs['batch'])

    while miss_streak < sargs['max_misses']:

        model_dir = os.path.join(sargs['sandbox'], 'run_' + str(rounds) + '_' + str(miss_streak))

        extra_args = {
            'class_num': CLASS_NUM,
            'columns': my_feature_columns,
            'steps_per_epoch': steps_per_epoch,
            'learning_rate': sargs['lr'] / (2 ** misses),
            'model_dir': model_dir,
            'warm_start_dir': best_model_dir,
            'feature_num' : train[0].shape[1]
        }
        merged_args = {**args, **sargs, **extra_args}

        # Create a new classifier instance
        classifier = cl.create_classifier(merged_args)

        # Train the model for exactly 1 epoch      
        classifier.fit(
            train_set,
            epochs = 1,
            batch_size = sargs['batch']
        )

        # Evaluate the model
        eval_result = classifier.evaluate(dev_set, batch_size = sargs['batch'])
        eval_result = results_to_dict(eval_result)
        log('Round ' + str(rounds) + '_' + str(miss_streak) + ', Fmes: ' + str(best_result['fmes']) + ' --> ' + str(eval_result['fmes']))
        if eval_result['fmes'] > best_result['fmes']:
            best_result = eval_result
            best_model_dir = model_dir
            best_classifier = classifier
            miss_streak = 0
            rounds += 1
            log('Improvement, go on...')
        else:
            miss_streak += 1
            misses += 1
            log('Miss #' + str(misses) + ', (streak = ' + str(miss_streak) + ')')
        
        # Cleanup sandbox not to run out of space due to models
        for m_dir in os.listdir(sargs['sandbox']):
            abs_m_dir = os.path.join(sargs['sandbox'], m_dir)
            if best_model_dir != abs_m_dir and model_dir != abs_m_dir:
                tf.summary.FileWriterCache.clear()
                shutil.rmtree(abs_m_dir)                

    final_result_train = best_classifier.evaluate(train_set, batch_size = sargs['batch'])
    final_result_dev = best_classifier.evaluate(dev_set, batch_size = sargs['batch'])
    final_result_test = best_classifier.evaluate(to_tf_dataset(test, sargs['batch']), batch_size = sargs['batch'])

    final_result_train = results_to_dict(final_result_train)
    final_result_dev = results_to_dict(final_result_dev)
    final_result_test = results_to_dict(final_result_test)

    return final_result_train, final_result_dev, final_result_test, best_classifier
       