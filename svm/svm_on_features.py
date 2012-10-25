import argparse
import copy
import numpy
import theano
import pickle
import logging
from pylearn2.utils import options_parser
from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.preprocessing import Standardize
from pylearn2.training_callbacks.training_callback import TrainingCallback
from scikits.learn.svm import LinearSVC
from pylearn2.models import svm


logging.basicConfig(level=logging.INFO)

class pylearn2_svm_callback(TrainingCallback):

    def __init__(self, **kwargs):
        self.svm_on_features = SVMOnFeatures(**kwargs)
    
    def __call__(self, model, dataset, algorithm):
        best_svm, valid_error, test_error = self.svm_on_features.run()
        i = model.batches_seen
        if not hasattr(model, 'results'):
            model.results = {}
        model.results['batches_seen'] = i
        model.results['valerr_%i' % i] = valid_error
        model.results['tsterr_%i' % i] = test_error
        model.results['C%i' % i] = best_svm.C


class SVMOnFeatures():
    
    def __init__(self, svm, trainset, testset,
                 model=None, model_call_kwargs=None,
                 validset=None, C_list=None, save_fname=None):
        """
        Performs cross-validation of a linear SVM on features extracted by a unsupervised
        learning module.

        Parameters
        ----------
        svm: sklearn.svm
            Scikit-learn SVM object to use for training.
        trainset: pylearn2.datasets.dataset.Dataset
            Pylearn2 dataset used for (feature extraction, SVM training).
        testset: pylearn2.datasets.dataset.Dataset
            Pylearn2 dataset used for final performance measurement. If no validation set is
            provided, the test will also be used for hyper-parameter selection.
        model: pylearn2.models.model.Model
            Unsupervised learning module used for feature extraction.
        model_call_kwargs: dict (optional)
            Dictionary of arguments to pass to model.call.
        validset: pylearn2.datasets.dataset.Dataset
            Pylearn2 dataset used for hyper-parameter selection.
        C_list: list
            List of penalty parameters to cross-validate for the SVM.
        save_fname: string
            Output filename to store trained svm model.
        """
        assert hasattr(model, 'perform')
        self.svm = svm
        self.trainset = trainset
        self.validset = validset
        self.testset = testset
        self.model = model
        self.model_call_kwargs = model_call_kwargs
        if C_list is None:
            C_list = [1e-3,1e-2,1e-1,1,10]
        self.C_list = C_list
        self.save_fname = save_fname

    def extract_features(self, dset, preproc=None, can_fit=False):
        new_dset = dense_design_matrix.DenseDesignMatrix()
        new_dset.X = self.model.perform(dset.X)
        new_dset.y = dset.y
        if preproc:
            preproc.apply(new_dset, can_fit=True)
        return new_dset
 
    def run(self, retrain_on_valid=False):
        # Extract features from model.
        preproc = Standardize()
        self.model.fn = self.model.function("perform", **self.model_call_kwargs)
        newtrain = self.extract_features(self.trainset, preproc, can_fit=True)
        newtest  = self.extract_features(self.testset, preproc, can_fit=False)
        newvalid = newtest if not self.validset else\
                   self.extract_features(self.validset, preproc, can_fit=False)
        # Find optimal SVM hyper-parameters.
        (best_svm, valid_error) = cross_validate_svm(self.svm,
                newtrain, newvalid, self.C_list)
        logging.info('Best validation error for C=%f : %f' % (best_svm.C, valid_error))
        # Optionally retrain on validation set, using optimal hyperparams.
        if self.validset and retrain_on_valid:
            retrain_svm(best_svm, newtrain, newvalid)
        test_error = compute_test_error(best_svm, newtest)
        logging.info('Test error = %f' % test_error)
        if self.save_fname:
            fp = open(self.save_fname, 'w')
            pickle.dump(best_svm, fp)
            fp.close()
        return (best_svm, valid_error, test_error)


import time
def cross_validate_svm(svm, trainset, validset, C_list):
    best_svm = None
    best_error = numpy.Inf
    print 'C_list = ', C_list
    for C in C_list:
        print 'C = ', C
        t1 = time.time()
        if hasattr(svm, 'set_params'):
            svm.set_params(C = C)
        else:
            svm.C = C
        svm.fit(trainset.X, trainset.y)
        predy = svm.predict(validset.X)
        error = (validset.y != predy).mean()
        if error < best_error:
            logging.info('SVM(C=%f): valid_error=%f **' % (C, error))
            best_error = error
            best_svm = copy.deepcopy(svm)
            # Numpy bug workaround: copy module does not respect C/F ordering.
            best_svm.raw_coef_ = numpy.asarray(best_svm.raw_coef_, order='F')
        else:
            logging.info('SVM(C=%f): valid_error=%f' % (C, error))
        logging.info('Elapsed time: %f' % (time.time() - t1))
    return (best_svm, best_error)


def retrain_svm(svm, trainset, validset):
    assert validset is not None
    logging.info('Retraining on {train, validation} sets.')
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
    obj.run(retrain_on_valid=False)
