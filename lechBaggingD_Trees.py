#!/usr/bin/python3
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', 'GTK3Agg', etc.
import matplotlib.pyplot as plt


def bagged_trees(X_train, y_train, X_test, y_test, num_bags, digit1, digit2):
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

    print("\n")
    print(f"BAGGED TREES for {digit1} vs {digit2}:")

    n_samples = X_train.shape[0]
    trees = []
    oob_predictions = np.zeros((n_samples, num_bags))
    for i in range(num_bags):
        # Bootstrap sampling
        sample_indices = np.random.choice(n_samples, n_samples, replace=True)
        oob_indices = np.setdiff1d(range(n_samples), sample_indices)
        X_sample, y_sample = X_train[sample_indices], y_train[sample_indices]

        # Train decision tree
        tree = DecisionTreeClassifier(criterion='entropy')
        tree.fit(X_sample, y_sample)
        trees.append(tree)

        # Out-of-bag prediction
        oob_predictions[oob_indices, i] = tree.predict(X_train[oob_indices])

    # Calculate out-of-bag error
    oob_error = np.mean((np.max(oob_predictions, axis=1) != y_train) * 1.0)

    # Test error calculation
    test_predictions = np.array([tree.predict(X_test) for tree in trees])
    test_prediction = np.mean(test_predictions, axis=0)
    test_error = np.mean(test_prediction != y_test)

    # TODO: Print the final OOB error for debugging
    print(f"Final OOB error: {oob_error}")

    #TODO: Remove Print the final test error for debugging
    print(f"Final test error: {test_error}")

    return oob_error, test_error

def single_decision_tree(X_train, y_train, X_test, y_test, digit1, digit2):
    print("\n")
    print(f"SINGLE DECISION TREE for {digit1} vs {digit2}:" )

    # Create a decision tree classifier object with entropy (information gain) as the criterion
    tree = DecisionTreeClassifier(criterion='entropy')

    # Train the decision tree on the training data
    tree.fit(X_train, y_train)

    # Predict on training data and calculate training error
    train_predictions = tree.predict(X_train)
    train_error = np.mean(train_predictions != y_train)

    # Predict on test data and calculate test error
    test_predictions = tree.predict(X_test)
    test_error = np.mean(test_predictions != y_test)

    # TODO: Print the final training error for debugging
    print(f"Final training error error: {train_error}")

    # TODO: Remove Print the final test error for debugging
    print(f"Final test error: {test_error}")

    return train_error, test_error

def split_data(first, second):
    # Step a1: Load the Data
    train_data = np.loadtxt('zip.train')
    test_data = np.loadtxt('zip.test')

    # Step a2: Filter the Data for Binary Classification
    # Let's say we're doing 1 vs. 3 first
    train_data_First_VS_Second = train_data[(train_data[:, 0] == first) | (train_data[:, 0] == second)]
    test_data_First_VS_Second = test_data[(test_data[:, 0] == first) | (test_data[:, 0] == second)]

    # Step a3: Split the Data into Features and Labels
    X_train_First_VS_Second = train_data_First_VS_Second[:, 1:]  # all columns except the first
    y_train_First_VS_Second = train_data_First_VS_Second[:, 0]  # first column is the label

    X_test_First_VS_Second = test_data_First_VS_Second[:, 1:]
    y_test_First_VS_Second = test_data_First_VS_Second[:, 0]

    # Convert labels to binary (1 for digit `first`, 0 for digit `second`)
    y_train_First_VS_Second = (y_train_First_VS_Second == first).astype(int)
    y_test_First_VS_Second = (y_test_First_VS_Second == first).astype(int)

    return X_train_First_VS_Second, y_train_First_VS_Second, X_test_First_VS_Second, y_test_First_VS_Second

def main():

    num_bags = 200

    # Split the training data for (1 VS 3) and (3 VS 5)
    X_train_1v3, y_train_1v3, X_test_1v3, y_test_1v3 = split_data(1, 3)
    X_train_3v5, y_train_3v5, X_test_3v5, y_test_3v5 = split_data(3, 5)

    # Run bagged trees for 1 VS 3
    out_of_bag_error_1v3, test_error_1v3 = bagged_trees(X_train_1v3, y_train_1v3, X_test_1v3, y_test_1v3, num_bags, 1, 3)

    # Run bagged trees for 3 VS 5
    out_of_bag_error_3v5, test_error_3v5 = bagged_trees(X_train_3v5, y_train_3v5, X_test_3v5, y_test_3v5, num_bags, 3, 5)

    # Run single decision tree for 1 VS 3
    train_error_1v3, test_error_1v3 = single_decision_tree(X_train_1v3, y_train_1v3, X_test_1v3, y_test_1v3, 1, 3)

    # Run single decision tree for 3 VS 5
    train_error_3v5, test_error_3v5 = single_decision_tree(X_train_3v5, y_train_3v5, X_test_3v5, y_test_3v5, 3, 5)


if __name__ == "__main__":
    main()

