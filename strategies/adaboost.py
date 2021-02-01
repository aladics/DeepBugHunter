import argparse
import dbh_util as util

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser()

parser.add_argument('--n-estimators', type=int, default=10, help='Maximum number of trees at which boosting is terminated.')
parser.add_argument('--learning-rate', type=float, default=1, help='Shrinks the contribution of each tree')
parser.add_argument('--max-depth', type=int, default = 1, help='Max depth of each tree')
parser.add_argument('--min-samples-leaf', type=int, default=1, help='Minimum number of samples at a leaf node')


def predict(classifier, test, args, sargs_str, threshold=None):
    sargs = util.parse(parser, sargs_str.split())
    preds = classifier.predict(test[0])
    if threshold is not None:
        preds = [1 if x >= threshold else 0 for x in preds]
    return preds


def learn(train, dev, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())
    base_classifier = DecisionTreeClassifier(max_depth = sargs.pop('max_depth'), min_samples_leaf = sargs.pop('min_samples_leaf'))
    return util.sklearn_wrapper(train, dev, test, AdaBoostClassifier(base_estimator = base_classifier, **sargs))