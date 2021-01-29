import argparse
import dbh_util as util

from sklearn.ensemble import AdaBoostClassifier

parser = argparse.ArgumentParser()

arser.add_argument('--n-estimators', type=int, default=10, help='Maximum number of estimators at which boosting is terminated.')
parser.add_argument('--learning-rate', type=int, default=1, help='Shrinks the contribution of each classifier')


def predict(classifier, test, args, sargs_str, threshold=None):
    sargs = util.parse(parser, sargs_str.split())
    preds = classifier.predict(test[0])
    if threshold is not None:
        preds = [1 if x >= threshold else 0 for x in preds]
    return preds


def learn(train, dev, test, args, sargs_str):
    sargs = util.parse(parser, sargs_str.split())
    return util.sklearn_wrapper(train, dev, test, AdaBoostClassifier(**sargs))