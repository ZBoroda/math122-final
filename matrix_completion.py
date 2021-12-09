from utils import *


class MatrixCompletionizationizer:

    def __init__(self, n_rows, n_cols, row_features=np.array([]), col_features=np.array([]), k=15):
        """Initalize a random model"""
        if row_features.size != 0:
            self.k = row_features.shape[1]
        elif col_features.size != 0:
            self.k = col_features.shape[1]
        else:
            self.k = k
        self.n_rows = n_rows
        self.n_cols = n_cols
        if row_features.size == 0:
            self.row_features = np.random.normal(size=(n_rows, k))
        else:
            self.row_features = row_features
            self.row_features -= self.row_features.mean(axis=1)[:, None]
            self.row_features /= self.row_features.std(axis=1)[:, None]
        if col_features.size == 0:
            self.col_features = np.random.normal(size=(n_cols, self.k))
        else:
            self.col_features = col_features
            self.col_features -= self.col_features.mean(axis=1)[:, None]
            self.col_features /= self.col_features.std(axis=1)[:, None]
        raw_predictions = self.predict()
        s = np.sqrt(2 * raw_predictions.std())  # We want to start out with roughly unit variance
        b = np.sqrt((5 - raw_predictions.mean() / s) / k)  # We want to start out with average rating 3.5
        self.row_features /= s
        self.row_features += b
        self.col_features /= s
        self.col_features += b

    def predict(self):
        """The model's predictions for all row/col pairs"""
        return self.row_features @ self.col_features.T

    def single_example_step(self, row, col, rating, learning_rate):
        """Update the model using the gradient at a single training example"""
        residual = np.dot(self.row_features[row], self.col_features[col]) - rating
        grad_rows = 2 * residual * self.col_features[col]  # the gradient for the row_features matrix
        grad_cols = 2 * residual * self.row_features[row]  # the gradient for the col_features matrix
        self.row_features[row] -= learning_rate * grad_rows
        self.col_features[col] -= learning_rate * grad_cols

    def train(self, train_rows, train_cols, train_vals,
              row_id_to_ind, col_id_to_ind,
              num_epochs=1000, batch_size=1000, learning_rate=0.005,
              print_loss=False, graph_loss=False,
              original_train_rows=np.array([]), original_train_cols=np.array([]), original_train_vals=np.array([]),
              original_row_id_to_ind=np.array([]), original_col_id_to_ind=np.array([])):
        """Train the model for a number of epochs"""
        # retrieve original data (before appending columns)
        if original_train_rows.size == 0:
            original_train_rows = train_rows
        if original_train_cols.size == 0:
            original_train_cols = train_cols
        if original_train_vals.size == 0:
            original_train_vals = train_vals
        if len(original_row_id_to_ind) == 0:
            original_row_id_to_ind = row_id_to_ind
        if len(original_col_id_to_ind) == 0:
            original_col_id_to_ind = col_id_to_ind
        original_train_rows_inds = np.array([original_row_id_to_ind[r] for r in original_train_rows])
        original_train_cols_inds = np.array([original_col_id_to_ind[r] for r in original_train_cols])
        # normalize the data
        train_vals_mean = train_vals.mean()
        train_vals_std = train_vals.std()
        train_vals_normalized = train_vals.copy()
        train_vals_normalized -= train_vals_mean
        train_vals_normalized /= train_vals_std
        # train the model
        train_MSEs = []
        m = len(train_vals_normalized)
        # It's good practice to shuffle your data before doing batch gradient descent,
        # so that each mini-batch performs like a random sample from the dataset
        for epoch in range(num_epochs):
            shuffle = np.random.permutation(m)
            shuffled_rows = train_rows.reset_index(inplace=False).loc[shuffle]['USER ID']
            shuffled_cols = train_cols.reset_index(inplace=False).loc[shuffle]['PRODUCT']
            shuffled_vals = train_vals_normalized.reset_index(inplace=False).loc[shuffle]['RATING']
            for row, col, val in zip(shuffled_rows[:batch_size], shuffled_cols[:batch_size],
                                     shuffled_vals[:batch_size]):
                # update the model using the gradient at a single example
                self.single_example_step(row_id_to_ind[row], col_id_to_ind[col], val, learning_rate)
            # after each Epoch, we'll evaluate our model
            predicted = self.predict()
            # unnormalize the data back and compute train loss
            predicted_unnormalized = predicted.copy()
            predicted_unnormalized *= train_vals_std
            predicted_unnormalized += train_vals_mean
            predicted_bounded = np.clip(predicted_unnormalized, 0, 10)
            train_loss = mse(original_train_vals,
                             predicted_bounded[original_train_rows_inds, original_train_cols_inds])
            if print_loss and (epoch == 0 or (epoch + 1) % (num_epochs / 10) == 0):
                print("Train loss after epoch #{} is: {}".format(epoch + 1, train_loss))
            train_MSEs.append(train_loss)
        predicted_unnormalized = predicted.copy()
        predicted_unnormalized *= train_vals_std
        predicted_unnormalized += train_vals_mean
        predicted_bounded = np.clip(predicted_unnormalized, 0, 10)
        if graph_loss:
            plt.scatter(np.array(range(len(train_MSEs))), np.array(train_MSEs), s=2, c='b', marker='o')
            plt.title("Train loss vs. iteration")
            plt.xlabel("Iteration #")
            plt.ylabel("Train MSE")
            plt.show()
        return predicted_bounded
