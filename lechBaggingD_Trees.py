#!/usr/bin/python3
# Homework 4 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def bagged_trees(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees 
    # and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function

    out_of_bag_error =
    test_error =

    return out_of_bag_error, test_error

def main_hw4():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_bags =

    # Split data
    X_train =
    y_train =
    X_test =
    y_test =

    # Run bagged trees
    out_of_bag_error, test_error = bagged_tree(X_train, y_train, X_test, y_test, num_bags)
    train_error, test_error = single_decision_tree(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main_hw4()

