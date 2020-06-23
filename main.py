"""
Main file to run the CNN_GRU model
    Two modes for training:
        AE: training with adversarial examples
        without_AE: regular training

    Within each mode of training there are two modes for model evaluation:
        cv: one seizure leave out cross validation
        test: devide the data to training evaluation and testing

"""

import tensorflow as tf
from utils.load_signals import PrepData
from utils.load_results import summary_results, load_results, auc_results
from models.model_ae import CNN_GRU
import os
import numpy as np
import sys
import warnings
import argparse


if not sys.warnoptions:
    warnings.simplefilter("ignore")


def data_loading(target, dataset,settings ):
    """ Extract the data from .edf files, prepare the data using PrepData class
        and then save it in cachedir

        Params:
                target (str) : number of patient
                dataset (str) : name of dataset to load

        Returns:
                Preprocessed EEG ictal and preictal data

    """

    print('Data Loading...................................')

    ictal_X, ictal_y = PrepData(target, type='ictal',
                                settings=settings).apply()
    interictal_X, interictal_y = PrepData(target, type='interictal',
                                          settings=settings).apply()

    return ictal_X, ictal_y, interictal_X, interictal_y, settings
    
    
def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Epilepsy dataset [CHBMIT or FB]', type=str, default='CHBMIT')
    parser.add_argument('-val', '--validation', help='Validation training mode [cv, test]', type=str, default='cv')
    parser.add_argument('-m', '--mode', help='Training mode augmentstion [AE, without_AE]', type=str, default='AE')
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=256)
    parser.add_argument('-e', '--epoch', help='Full training epochs', type=int, default=50)
    parser.add_argument('-p', '--percentage', help='Percentage of the AE of the total data to generate', type=int, default=40)
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    return parser.parse_args()


def train(args):
    dataset = args.dataset
    if dataset == 'CHBMIT':
        patients = [
                 '1',
                 '2',
                 '3',
                 '5',
                 '9',
                 '10',
                 '13',
                 '14',
                 '18',
                 '19',
                 '20',
                 '21',
                 '23'
            ]
    elif dataset == 'FB':
        patients = [
                 '1',
                 '3',
                 '4',
                 '5',
                 '6',
                 '14',
                 '15',
                 '16',
                 '17',
                 '18',
                 '19',
                 '20',
                 '21'
             ]


    settings = {"dataset": dataset,
                "datadir": dataset,
                "cachedir": "{}_cache".format(dataset),
                "results": "results\\results_{}\\".format(dataset),
                "resultsCV": "results\\results_CV_{}\\".format(dataset),
                "resultsCV_AE": "results\\resultsCV_{}_AE\\".format(dataset),
                "results_AE": "results\\results_{}_AE\\".format(dataset)}
    
    for i in range(len(patients)):

        target = patients[i]
        # loading the data for each patient
        ictal_X, ictal_y, interictal_X,\
            interictal_y, settings = data_loading(target, dataset, settings)
        # Resetting the graph
        tf.reset_default_graph()
        graph = tf.get_default_graph()
        session = tf.Session()
        noise_limit = 0.3   #limit of the max value for generated AE examples
        model = CNN_GRU([ictal_X.shape[2], ictal_X.shape[3]],
                        dataset, noise_limit, graph)   # Build the graph
        model.train_eval_test_ae(
                session, target, ictal_X, ictal_y, interictal_X,
                interictal_y, settings, validation=args.validation, 
                mode=args.mode, batch_size=args.batch_size, epoch=args.epoch, 
                percentage=args.percentage, verbose=args.verbose)
    
        session.close()
    return settings


def main():
    args = get_args()
    settings = train(args)
    
    #print the results
    print("\n")
    print('************ Final Results on {} Dataset *********************'.format(args.dataset))
    if args.validation == 'cv':
        results_path = settings['resultsCV']
    elif args.validation == 'cv' and args.mode == 'AE':
        results_path = settings['resultsCV_AE']
    elif args.validation == 'test':
        results_path = settings['results']
    elif args.validation == 'test' and args.mode == 'AE':
        results_path = settings['results_AE']
        
    os.makedirs(results_path, exist_ok=True) 
    data_results, patients=load_results(results_path, args.dataset)
    summary_results(patients, data_results)
    print("AVG_AUC: ", np.mean(auc_results(data_results, patients)))
    
if __name__ == "__main__":
    main()

