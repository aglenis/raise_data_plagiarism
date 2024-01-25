import pandas as pd
import numpy as np

from pyts.transformation import WEASEL

from sklearn.metrics.pairwise import cosine_similarity


from pyts.approximation import SymbolicFourierApproximation
from pyts.utils.utils import _windowed_view

from sklearn.feature_extraction.text import CountVectorizer

from itertools import product

import matplotlib.pyplot as plt

import random

def do_symbolic(X,window_size,window_step):

    drop_sum=False
    norm_mean=False
    norm_std=False

    word_size = 4
    n_bins = 4



    n_timestamps = X.shape[1]
    n_samples = X.shape[0]

    y= [i for i in range(n_samples)]

    n_windows = ((n_timestamps - window_size + window_step)
                         // window_step)

    X_windowed = _windowed_view(
        X, n_samples, n_timestamps, window_size, window_step
    )
    X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)


    sfa = SymbolicFourierApproximation(
        n_coefs=word_size, drop_sum=False,
        anova=False, norm_mean=norm_mean,
        norm_std=norm_std, n_bins=n_bins,
        strategy='normal', alphabet=None
    )
    y_repeated = np.repeat(y, n_windows)
    X_sfa = sfa.fit_transform(X_windowed, y_repeated)


    X_word = np.asarray([''.join(X_sfa[i])
                         for i in range(n_samples * n_windows)])
    X_word = X_word.reshape(n_samples, n_windows)

    X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])

    prod = product(['a','b','c','d'],repeat=4)
    prod_list = list(prod)
    vocabulary = []
    for curr_prod in prod_list:
        curr_str = "".join(curr_prod)
        vocabulary.append(curr_str)


    vectorizer = CountVectorizer(ngram_range=(1, 1),vocabulary=vocabulary)
    X_counts = vectorizer.fit_transform(X_bow)

    return X_counts

def return_cosine_similarity(numpy_array,numpy_array_new):
    window_size_arg=12
    transformed = do_symbolic(numpy_array,window_size=window_size_arg,window_step=1)
    transformed_average = np.mean(transformed,axis=0)

    transformed_numpy_array_new = do_symbolic(numpy_array_new,window_size=window_size_arg,window_step=1)
    transformed_average_new = np.mean(transformed_numpy_array_new,axis=0)
    curr_cosine = cosine_similarity(transformed_average,transformed_average_new)[0][0]

    return curr_cosine


def create_noise(numpy_array,coeff):
    noise = np.random.normal(1,coeff,numpy_array.shape[1])

    final_numpy = numpy_array*noise

    return final_numpy


def compute_plagiarism_metrics(numpy_array,coeff):

    np.random.seed(42)
    random.seed(42)

    noise_signal = create_noise(numpy_array,coeff)

    n_rows = numpy_array.shape[0]
    n_cols = numpy_array.shape[1]

    num_rows_random = int(coeff*n_rows)
    rows_indexes = random.sample(range(0,n_rows),num_rows_random)

    numpy_deleted_rows = np.delete(numpy_array, rows_indexes, axis=0)
    numpy_appended_rows = np.vstack([numpy_array,numpy_array[rows_indexes,:]])


    num_cols_random = int(coeff*n_cols)
    cols_indexes = random.sample(range(0,n_cols),num_cols_random)

    numpy_deleted_cols = np.delete(numpy_array, cols_indexes, axis=1)
    numpy_appended_cols = np.hstack([numpy_array,numpy_array[:,cols_indexes]])


    noise_similarity = return_cosine_similarity(numpy_array,noise_signal)

    appended_rows_similarity = return_cosine_similarity(numpy_array,numpy_appended_rows)
    deleted_rows_similarity = return_cosine_similarity(numpy_array,numpy_deleted_rows)

    appended_cols_similarity = return_cosine_similarity(numpy_array,numpy_appended_cols)
    deleted_cols_similarity = return_cosine_similarity(numpy_array,numpy_deleted_cols)

    return [noise_similarity,appended_rows_similarity,deleted_rows_similarity,appended_cols_similarity,deleted_cols_similarity]


def create_plot(x_axis,y_axis,title_name,xname,yname):

    plt.plot(x_axis, y_axis)
    plt.title(title_name)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()

def process_dataset(dataset_name,dataset_array,coeffs):

    dataset_list = []
    for curr_coeff in coeffs:
        #print(curr_coeff)
        ret_list = compute_plagiarism_metrics(dataset_array,curr_coeff)
        #print(ret_list)
        dataset_list.append(ret_list)

    #print(dataset_list)

    plot_array = np.array(dataset_list)

    # [noise_similarity,appended_rows_similarity,deleted_rows_similarity,appended_cols_similarity,deleted_cols_similarity]

    print(plot_array)

    columns_arg = ['Noise similarity','Rows append similarity','Rows delete similarity','Columns append similarity','Columns delete similarity']
    results_dataframe = pd.DataFrame(columns=columns_arg,data=plot_array)
    results_dataframe['percentage'] = coeffs

    results_dataframe.to_csv(dataset_name+'.csv',index=False)
