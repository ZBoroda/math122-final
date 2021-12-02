# math122-final
math122-final

Stuff to do:

- [ ] Complete utility functions:
  - [ ] Split ratings into testing and training data
    - [ ] Normalize training data 
  - [ ] Write a matrix compeltion function
  - [ ] Write a function that calculates testing and training error 
- [ ] First aproach: Concatonate the two matricies and do matrix completion with multiple types of user history features
  - [ ] The Entire matrix of user features 
  - [ ] The PCA of the features matrix 
  - [ ] The user features clustered and then PCA on each cluster 
- [ ] Second aproach: Treat the user history as 1/2 of the Matrix completion problem (ie. the normalized user history is the P matrix)
  - [ ] The Entire matrix of user features 
  - [ ] The PCA of the features matrix 
  - [ ] The user features clustered and then PCA on each cluster
- [ ] Pick which ever one of the two aproaches is better by testing accuracy
- [ ] Use the data from the best aproach on a Matrix completion problem for the entire 4500 row problem with either aproach 1 or 2
