#!/usr/bin/python3
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', 'GTK3Agg', etc.
import matplotlib.pyplot as plt


def bagged_trees(X_train, y_train, X_test, y_test, max_num_bags, digit1, digit2):
    print("\n")
    print(f"BAGGED TREES for {digit1} vs {digit2}:")

    n_samples = X_train.shape[0]
    trees = []
    sample_indices_list = []  # List to store the indices used for each bag
    oob_errors = []  # List to store OOB errors for different numbers of bags

    for num_bags in range(1, max_num_bags + 1):
        if num_bags > len(trees):
            # Bootstrap sampling
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            sample_indices_list.append(sample_indices)
            X_sample, y_sample = X_train[sample_indices], y_train[sample_indices]

            # Train decision tree
            tree = DecisionTreeClassifier(criterion='entropy')
            tree.fit(X_sample, y_sample)
            trees.append(tree)

            # Initialize arrays to accumulate OOB predictions
            oob_predictions_sum = np.zeros(n_samples)
            oob_predictions_count = np.zeros(n_samples)

            for tree_idx, tree in enumerate(trees):
                # Find the out-of-bag samples for this tree
                oob_indices = np.setdiff1d(range(n_samples), sample_indices_list[tree_idx])

                # Accumulate OOB predictions
                oob_predictions_sum[oob_indices] += tree.predict(X_train[oob_indices])
                oob_predictions_count[oob_indices] += 1

                # Avoid division by zero for samples that were never out-of-bag
            oob_predictions_count[oob_predictions_count == 0] = 1

            # Calculate the average OOB prediction for each sample
            oob_predictions_avg = oob_predictions_sum / oob_predictions_count

            # Binarize the predictions (since this is a binary classification task)
            oob_predictions_avg = (oob_predictions_avg > 0.5).astype(int)

            # Calculate OOB error
            oob_error = np.mean(oob_predictions_avg != y_train)
            oob_errors.append(oob_error)

            # Final out-of-bag error for the maximum number of bags
            final_oob_error = oob_errors[-1]

            # Test error calculation
            test_predictions = np.array([tree.predict(X_test) for tree in trees])
            test_prediction = np.mean(test_predictions, axis=0)
            test_error = np.mean(test_prediction != y_test)

    # Final out-of-bag error for the maximum number of bags
    final_oob_error = oob_errors[-1]

    # Test error calculation
    test_predictions = np.array([tree.predict(X_test) for tree in trees])
    test_prediction = np.mean(test_predictions, axis=0)
    test_error = np.mean(test_prediction != y_test)

    print(f"Final OOB error for {max_num_bags} bags: {final_oob_error}")
    print(f"Final test error: {test_error}")

    return oob_errors, final_oob_error, test_error

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
    out_of_bag_errors_1v3,final_out_of_bag_errors_1v3, test_error_1v3 = bagged_trees(X_train_1v3, y_train_1v3, X_test_1v3, y_test_1v3, num_bags, 1, 3)

    # Run bagged trees for 3 VS 5
    out_of_bag_errors_3v5, final_out_of_bag_errors_3v5 , test_error_3v5 = bagged_trees(X_train_3v5, y_train_3v5, X_test_3v5, y_test_3v5, num_bags, 3, 5)

    # Run single decision tree for 1 VS 3
    train_error_1v3, test_error_1v3 = single_decision_tree(X_train_1v3, y_train_1v3, X_test_1v3, y_test_1v3, 1, 3)

    # Run single decision tree for 3 VS 5
    train_error_3v5, test_error_3v5 = single_decision_tree(X_train_3v5, y_train_3v5, X_test_3v5, y_test_3v5, 3, 5)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 201), out_of_bag_errors_1v3, marker='o')
    plt.xlabel('Number of Bags')
    plt.ylabel('OOB Error')
    plt.title('OOB Error vs. Number of Bags (1 vs 3 Classification)')
    plt.grid(True)
    plt.show()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 201), out_of_bag_errors_3v5, marker='o')
    plt.xlabel('Number of Bags')
    plt.ylabel('OOB Error')
    plt.title('OOB Error vs. Number of Bags (3 vs 5 Classification)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

