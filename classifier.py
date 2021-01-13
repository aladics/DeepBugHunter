import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics as tf_metrics

import metrics as mt

def create_classifier(args):
    input = tf.keras.layers.Input(shape=(args['feature_num'],), batch_size=args['batch'])
    x = tf.keras.layers.Dense(args['neurons'], activation = 'relu')(input)
    for _ in range(args['layers']-1):
        x = tf.keras.layers.Dense(args['neurons'], activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(args['beta']))(x)
    x =  tf.keras.layers.Dense(1, activation = 'sigmoid')(x)


    #TODO: Add mt.mcc and mt.fmes
    eval_metrics = [tf_metrics.BinaryAccuracy(), tf_metrics.Precision(), tf_metrics.Recall(),
                    tf_metrics.TruePositives(), tf_metrics.TrueNegatives(), tf_metrics.FalsePositives(),
                    tf_metrics.FalseNegatives()]

    model = tf.keras.models.Model(inputs = input, outputs = x)
    model.compile(loss ='binary_crossentropy', optimizer = tf.keras.optimizers.Adagrad(args['learning_rate']), metrics = eval_metrics)

    return model
