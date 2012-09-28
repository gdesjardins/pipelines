import argparse
import copy
import numpy
import theano
import pickle
import logging
from pylearn2.utils import options_parser
from pylearn2.utils import serial
from pylearn2.datasets.preprocessing import Standardize
from scikits.learn.svm import LinearSVC


logging.basicConfig(level=logging.INFO)


class SVMOnFeatures():
    
    def __init__(self, model, svm, trainset, testset, validset=None,
                 C_list=None, save_fname=None, model_call_kwargs=None,
                 standardize=False):
        """
        Performs cross-validation of a linear SVM on features extracted by a unsupervised
        learning module.

        Parameters
        ----------
        model: pylearn2.models.model.Model
            Unsupervised learning module used for feature extraction.
        svm: sklearn.svm
            Scikit-learn SVM object to use for training.
        trainset: pylearn2.datasets.dataset.Dataset
            Pylearn2 dataset used for (feature extraction, SVM training).
        testset: pylearn2.datasets.dataset.Dataset
            Pylearn2 dataset used for final performance measurement. If no validation set is
            provided, the test will also be used for hyper-parameter selection.
        validset: pylearn2.datasets.dataset.Dataset
            Pylearn2 dataset used for hyper-parameter selection.
        C_list: list
            List of penalty parameters to cross-validate for the SVM.
        save_fname: string
            Output filename to store trained svm model.
        model_call_kwargs: dict (optional)
            Dictionary of arguments to pass to model.call.
        standardize: boolean
            Standardize the model output, before feeding to SVM.
        """
        assert hasattr(model, 'perform')
        self.model = model
        self.svm = svm
        self.trainset = trainset
        self.validset = validset
        self.testset = testset
        if C_list is None:
            C_list = [1e-3,1e-2,1e-1,1,10]
        self.C_list = C_list
        self.save_fname = save_fname
        self.model_call_kwargs = model_call_kwargs

    def extract_features(self):
        postproc = Standardize()
        self.model.fn = self.model.function("perform", self.model_call_kwargs)
        # extract new representation for training data
        self.trainset.X = self.model.perform(self.trainset.X)
        postproc.apply(self.trainset, can_fit=True)
        # extract new representation for test data
        self.testset.X = self.model.perform(self.testset.X)
        postproc.apply(self.testset, can_fit=False)
        # extract new representation for validation data
        if self.validset:
            self.validset.X = self.model.perform(self.validset.X)
            postproc.apply(self.validset, can_fit=False)
 
    def run(self, retrain_on_valid=True):
        self.extract_features()
        validset = self.validset if self.validset else self.testset
        self.svm = cross_validate_svm(self.svm, self.trainset, validset, self.C_list)
        if self.validset and retrain_on_valid:
            retrain_svm(self.svm, self.trainset, self.validset)
        test_error = compute_test_error(self.svm, self.testset)
        logging.info('Test error = %f' % test_error)
        if self.save_fname:
            fp = open(self.save_fname, 'w')
            pickle.dump(self.svm, fp)
            fp.close()
        return self.svm


def cross_validate_svm(svm, trainset, validset, C_list):
    best_svm = None
    best_error = numpy.Inf
    for C in C_list:
        svm.C = C
        svm.fit(trainset.X, trainset.y)
        predy = svm.predict(validset.X)
        error = (validset.y != predy).mean()
        if error < best_error:
            logging.info('SVM(C=%f): valid_error=%f **' % (C, error))
            best_error = error
            best_svm = copy.copy(svm)
        else:
            logging.info('SVM(C=%f): valid_error=%f' % (C, error))
    return best_svm


def retrain_svm(svm, trainset, validset):
    assert validset is not None
    full_train_X = numpy.vstack((trainset.X, validset.X))
    full_train_y = numpy.hstack((trainset.y, validset.y))
    svm.fit(full_train_X, full_train_y)
    return svm


def compute_test_error(svm, testset):
    test_predy = svm.predict(testset.X)
    return (testset.y != test_predy).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Train a linear SVM on the features extracted by a given model.")
    parser.add_argument('config', action='store', choices=None,
                        help='A YAML configuration file specifying the training procedure')
    args = parser.parse_args()
    obj = serial.load_train_file(args.config)
    obj.run()
