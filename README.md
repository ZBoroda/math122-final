# MATH122a Final Project: Predicting Product Ratings From Web Browsing Data
## Zachary Boroda, Julia Dai, Zuhaeer Islam, David Mehlhop, Ruth Rosenblum

Our code, contained in ```predict_ratings.ipynb```, can be run as a Jupyter notebook. The notebook references the following external files:<br>
- ```user_history.csv```
- ```user_ratings.csv```
- ```matrix_completion.py```
  - This file includes our ```MatrixCompletionizationizer``` object, which can be initialized and then trained using stochastic gradient descent.
- ```utils.py```
  - This file includes the following functions:
    - ```mse```: computes the Mean Squared Error given predicted and actual ratings
    - ```split_data```: splits the given data into a train and test subset
    - ```PCA```: performs Principal Component Analysis on the given data
    - ```KMeansPCA```: clusters the given data using the K-means algorithm, and then performs PCA on each cluster individually

Run as is, the code will first train our chosen model with the optimal hyperparameters on the training subset of the data, printing the train MSE after every 90 epochs and the test MSE and average error.
Then, it will re-train the same model (with the same hyperparameters), generate the predictions, and save them in ```predictions.csv```.

In order to run the other models described in our report, one can modify the values in the second cell of the Jupyter notebook:
- ```FEATURE_SELECTION```: 
  - 0 corresponds to using the entire web history matrix as features
  - 1 corresponds to using the reduced-dimensionality web history matrix
  - 2 corresponds to using the reduced-dimensionality clustered web history matrix with 
  - 3 corresponds to running all three methods
- ```MATRIX_COMPLETION```: 
  - 0 corresponds to using the web history features as initial feature vectors
  - 1 corresponds to appending the web history features to the ratings data
- ```learning_rate```: we tried values of 0.001, 0.002, 0.005, 0.01
- ```num_epochs```: we tried values of 500, 800, 900, 1000, 2000
- ```batch_size```: we tried values of 10, 100, 300, 500, 1000, 10000 
- ```k``` (only relevant if appending the web history features to the ratings data): we tried values of 15, 20, 30
