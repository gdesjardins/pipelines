import argparse
import copy
import numpy
import theano
import pickle
import logging
import time
from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.preprocessing import Standardize
from pylearn2.training_callbacks.training_callback import TrainingCallback
from logistic_cg import cg_optimization

logging.basicConfig(level=logging.INFO)

class pylearn2_svm_callback(TrainingCallback):

    def __init__(self, run_every, results_prefix='', retrain_on_valid=True, **kwargs):
        self.run_every = run_every
        self.retrain_on_valid = retrain_on_valid
        self.results_prefix = results_prefix
        self.log_reg_on_features = LogRegOnFeatures(**kwargs)
    
    def __call__(self, model, dataset, algorithm):
        if model.batches_seen == 0 or (model.batches_seen % self.run_every) != 0:
            return

        i = model.batches_seen
        best_svm, valid_error, test_error = self.svm_on_features.run(
                retrain_on_valid = self.retrain_on_valid)
        i = model.batches_seen

        results = {}
        results['%sbatches_seen' % self.results_prefix] = i
        results['%svalerr_%i' % (self.results_prefix, i)] = valid_error
        results['%ststerr_%i' % (self.results_prefix, i)] = test_error

        fp = open('svm_callback.log', 'a')
        fp.write('Batches seen: %i' % i)
        fp.write('\t best validation error: %f' % valid_error)
        fp.write('\t best test error: %f' % test_error)
        fp.write('\n')
        fp.close()

        if hasattr(model, 'jobman_channel') and model.jobman_channel:
            model.jobman_state.update(results)
            model.jobman_channel.save()

def is_onehot(y):
    return y.ndim == 2 and (numpy.sum(y, axis=1) == 1).all()

def process_labels(y):
    if is_onehot(y):
        return numpy.argmax(y, axis=1)
    else:
        return y.flatten()

class LogRegOnFeatures():
   
    def __init__(self, trainset, testset,
                 model=None, model_call_kwargs=None,
                 validset=None,
                 n_epoch_list=None,
                 save_fname=None):
        """
        Performs cross-validation of a linear SVM on features extracted by a unsupervised
        learning module.

        Parameters
        ----------
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
        n_epoch_list : list
            List of number of epoch to cross-validate for the model
        save_fname: string
            Output filename to store trained svm model.
        """
        self.trainset = trainset
        self.validset = validset
        self.testset = testset
        self.model = model
        #self.model.do_theano()
        self.model_call_kwargs = model_call_kwargs
        if n_epoch_list is None:
            n_epoch_list = [10, 25, 50, 100, 250]
        self.n_epoch_list = n_epoch_list
        self.save_fname = save_fname
        self.trainset_y = process_labels(self.trainset.y)
        self.validset_y = process_labels(self.validset.y)
        self.testset_y  = process_labels(self.testset.y)

    def extract_features(self, dset, preproc=None, can_fit=False):
        new_dset = dense_design_matrix.DenseDesignMatrix()
        if  str(self.model.__class__).find('DBM') != -1:
            self.model.set_batch_size(len(dset.X))
            self.model.setup_pos_func(dset.X)
            self.model.pos_func()
            new_dset.X = self.model.fn(dset.X)
        elif self.model:
            outsize = self.model.fn(dset.X[:1]).shape[1]
            X = numpy.zeros((len(dset.X), outsize))
            for i in xrange(0, len(X), self.model.batch_size):
                batch = dset.X[i : i + self.model.batch_size]
                X[i : i + len(batch)] = self.model.fn(batch)
            new_dset.X = X
        else:
            new_dset.X = dset.X

        if preproc:
            preproc.apply(new_dset, can_fit=True)

        return new_dset
 
    def run(self, retrain_on_valid=True):
        # Extract features from model.
        preproc = Standardize()
        self.model.fn = self.model.function("perform", **self.model_call_kwargs)
        newtrain = self.extract_features(self.trainset, preproc, can_fit=True)
        newtest  = self.extract_features(self.testset, preproc, can_fit=False)
        newvalid = newtest if not self.validset else\
                   self.extract_features(self.validset, preproc, can_fit=False)
                   
            
        # Find the best number of training epochs
        best_nb_epoch, valid_error = cross_validate_logistic_regression(
                                         (newtrain.X, self.trainset_y),
                                         (newtest.X, self.testset_y),
                                         self.n_epoch_list)
        logging.info('Best validation error for n_epoch=%i : %f' % (best_nb_epoch, valid_error))

        # Measure test error with the optimal number of epochs
        # (retraining on train and valid if applicable)
        if self.validset and retrain_on_valid:
            full_train_X = numpy.vstack((newtrain.X, newvalid.X))
            full_train_Y = numpy.hstack((self.trainset_y, self.validset_y))
        else:
            full_train_X = newtrain.X
            full_train_Y = self.trainset_y
        full_test_X = newtest.X
        full_test_Y = self.testset_y
            
        best_params, test_error = test_logistic_regression((full_train_X, full_train_Y),
                                                           (full_test_X, full_test_Y),
                                                           best_nb_epoch)
            
        logging.info('Test error = %f' % test_error)
        if self.save_fname:
            fp = open(self.save_fname, 'w')
            pickle.dump(best_params, fp)
            fp.close()
        return (best_params, valid_error, test_error)


def cross_validate_logistic_regression((train_X, train_y), (valid_X, valid_y),
                                       n_epoch_list):
                                           
    # Choose the best number of epochs for training
    best_error = numpy.Inf
    best_n_epoch = 0
    for n_epoch in n_epoch_list:
        t1 = time.time()
        
        best_model, valid_error = cg_optimization(n_epoch,
                                                  (train_X, train_y),
                                                  (valid_X, valid_y))                    
        
        if valid_error < best_error:
            best_error = valid_error
            best_n_epoch = n_epoch
            
        logging.info('LogisticRegression(n_epoch=%i): valid_error=%f' % (n_epoch, valid_error))
        logging.info('Elapsed time: %f' % (time.time() - t1))
        
    return best_n_epoch, best_error
        
               
def test_logistic_regression((train_X, train_y), (test_X, test_y),
                             n_epoch):
                                 
    best_model, error = cg_optimization(n_epoch,
                                        (train_X, train_y),
                                        (test_X, test_y))
    return best_model, error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Train a linear SVM on the features extracted by a given model.")
    parser.add_argument('config', action='store', choices=None,
                        help='A YAML configuration file specifying the training procedure')
    args = parser.parse_args()
    obj = serial.load_train_file(args.config)
    obj.run(retrain_on_valid=True)
