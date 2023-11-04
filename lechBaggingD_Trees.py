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

    # Initialize variables
    bagged_trees = []
    oob_predictions = np.zeros((X_train.shape[0], num_bags))
    oob_counts = np.zeros(X_train.shape[0])

    # List of OOB errors for each number of bags
    oob_errors = []

    # Bootstrap sampling and training decision trees
    for b in range(num_bags):
        bootstrap_indices = np.random.choice(range(X_train.shape[0]), size=X_train.shape[0], replace=True)
        oob_indices = np.setdiff1d(range(X_train.shape[0]), bootstrap_indices)

        #TODO: Remove Print the indices for debugging
        print(f"Bag {b + 1}:")
        print(f"Bootstrap indices (first 10): {bootstrap_indices[:10]}")
        print(f"OOB indices (first 10): {oob_indices[:10]}")

        # Train decision tree on bootstrap sample
        tree = DecisionTreeClassifier()
        tree.fit(X_train[bootstrap_indices], y_train[bootstrap_indices])
        bagged_trees.append(tree)

        # Out-of-bag predictions
        oob_predictions[oob_indices, b] = tree.predict(X_train[oob_indices])
        oob_counts[oob_indices] += 1

        #TODO: Remove Print the OOB predictions for debugging
        print(f"OOB predictions (first 10) for bag {b + 1}: {oob_predictions[oob_indices[:10], b]}")
        print(f"OOB counts (first 10): {oob_counts[oob_indices[:10]]}")

        # Calculate cumulative out-of-bag error after each bag
        if oob_counts[oob_indices].any():  # Avoid division by zero
            oob_error = (y_train[oob_indices] != np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), minlength=2).argmax(), 1,
                oob_predictions[oob_indices, :b + 1])) & (oob_counts[oob_indices] > 0)
            current_oob_error = np.sum(oob_error) / np.sum(oob_counts[oob_indices] > 0)
            oob_errors.append(current_oob_error)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_bags + 1), oob_errors, label='OOB Error Rate')
    plt.xlabel('Number of Bags')
    plt.ylabel('OOB Error Rate')
    plt.title(f'OOB Error Rate for Bagging Decision Trees ({digit1} vs {digit2})')
    plt.legend()
    plt.show()

    # Calculate final out-of-bag error
    final_oob_error = (y_train != np.apply_along_axis(lambda x: np.bincount(x.astype(int), minlength=2).argmax(), 1,
                                                      oob_predictions)) & (oob_counts > 0)
    out_of_bag_error = np.sum(final_oob_error) / np.sum(oob_counts > 0)

    #TODO: Print the final OOB error for debugging
    print(f"Final OOB error: {out_of_bag_error}")

    # Calculate test error
    test_votes = np.array([tree.predict(X_test) for tree in bagged_trees])
    test_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int), minlength=2).argmax(), 0, test_votes)
    test_error = np.mean(test_predictions != y_test)

    #TODO: Remove Print the final test error for debugging
    print(f"Final test error: {test_error}")

    return out_of_bag_error, test_error

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

    # Convert labels to binary (1 for digit First, 0 for digit Second)
    y_train_First_VS_Second = (y_train_First_VS_Second == 1).astype(int)
    y_test_First_VS_Second = (y_test_First_VS_Second == 1).astype(int)

    return X_train_First_VS_Second, y_train_First_VS_Second, X_test_First_VS_Second, y_test_First_VS_Second

def main():

    num_bags = 200

    # Split the training data for (1 VS 3) and (3 VS 5)
    X_train_1v3, y_train_1v3, X_test_1v3, y_test_1v3 = split_data(1, 3)
    X_train_3v5, y_train_3v5, X_test_3v5, y_test_3v5 = split_data(3, 5)

    # Run bagged trees for 1 VS 3
    out_of_bag_error_1v3, test_error_1v3 = bagged_trees(X_train_1v3, y_train_1v3, X_test_1v3, y_test_1v3, num_bags, 1, 3)

    # Run bagged trees for 3 VS 5
    # out_of_bag_error_3v5, test_error_3v5 = bagged_trees(X_train_3v5, y_train_3v5, X_test_3v5, y_test_3v5, num_bags, 3, 5)

    # train_error, test_error = single_decision_tree(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()

