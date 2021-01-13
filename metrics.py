import tensorflow as tf
import tensorflow.keras.metrics as tf_metrics 

# Matthew's correlation coefficient
def mcc(labels, preds):
    tp_m = tf_metrics.TruePositives()
    fp_m = tf_metrics.FalsePositives()
    tn_m = tf_metrics.TrueNegatives()
    fn_m = tf_metrics.FalseNegatives()

    tp_m.update_state(labels, preds)
    tn_m.update_state(labels, preds)
    fp_m.update_state(labels, preds)
    fn_m.update_state(labels, preds)
    
    tp = tp_m.result()
    tn = tn_m.result()
    fp = fp_m.result()
    fn = fn_m.result()
    eps = 1e-7

    num = (tp * tn - fp * fn)
    denom = tf.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) + eps

    return num / denom

# F-Measure
def fmes(labels, preds):
    precision_m =  tf_metrics.Precision()
    recall_m = tf_metrics.Recall()

    precision_m.update_state(labels, preds)
    recall_m.update_state(labels, preds)
    
    precision = precision_m.result()
    recall = recall_m.result()

    """ precision, precision_update = tf.compat.v1.metrics.precision(labels, preds)
    recall, recall_update = tf.compat.v1.metrics.recall(labels, preds) """
    eps = 1e-7

    num = (2 * precision * recall)
    denom = precision + recall + eps

    return num / denom