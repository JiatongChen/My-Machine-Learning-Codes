############ classification neural network ##############
NN_clf <- function(X_train, Y_train,  num_hidden = 20, 
                  num_iterations = 10000, learning_rate = 1e-1){
  ## num_hidden: Number of hidden units in the hidden layer
  ## num_iterations: Number of iterations.
  ## learning_rate: Learning rate.
  n = dim(X_train)[1]
  p = dim(X_train)[2] + 1
  X_train1 = cbind(rep(1,n), X_train)
  alpha = matrix(rnorm(p*num_hidden), nrow=p)
  beta = matrix(rnorm((num_hidden + 1)), nrow = num_hidden + 1)
  
  for (it in 1:num_iterations){
    # feed-forward
    Z = 1 / (1+exp(-X_train1 %*% alpha))
    Z1 = cbind(rep(1,n), Z)
    pr = 1 / (1+exp(-Z1 %*% beta))
    
    # error back-propagate
    dbeta =  t(Y_train-pr) %*% Z1
    beta = beta +learning_rate * t(dbeta)
    for(k in 1:num_hidden)
    {
      da = (Y_train-pr) *beta[k+1] *Z[,k]*(1-Z[,k])
      dalpha = matrix(rep(1,n), nrow=1) %*% ((matrix(da, n, p)*X_train1))/n
      alpha[,k] = alpha[,k] + learning_rate *t(dalpha)
    }

  }

  return(list(alpha = alpha, beta = beta))
  # Output:The learned weights of the input layer (alpha) and the hidden layer (beta)
}

pred_NN_clf <- function(NN_classifier, X_test){
  alpha = NN_classifier$alpha
  beta = NN_classifier$beta
  n = dim(X_test)[1]
  
  X_test1 = cbind(rep(1,n), X_test)
  Z_test = 1 / (1+exp(-X_test1 %*% alpha))
  
  Z1_test = cbind(rep(1,n), Z_test)
  pr = 1 / (1+exp(-Z1_test %*% beta))
  return(pr)
}

############### Classification Example ###############
# training data
set.seed(200)
n = 1000
X = matrix(runif(2*n), nrow = n)
Y_clf = (X[,1]^2 + X[,2]^2) <= 1
Y_clf = as.matrix(Y_clf)


# testing data
set.seed(150)
n_test = 1000
X_test = matrix(runif(2*n_test), nrow = n_test)
Y_clf_test = (X_test[,1]^2 + X_test[,2]^2) <= 1
Y_clf_test = as.matrix(Y_clf_test)

# visualize training data
plot(X[,1],X[,2],col=Y_clf+3,xlim = c(0,1), ylim = c(0,1))
# visualize testing data
plot(X_test[,1],X_test[,2],col=Y_clf_test+3,xlim = c(0,1), ylim = c(0,1))

# train the feed forward neural network
nn_classifier = NN_clf(X, Y_clf)

# prediction on training set
pr_train = pred_NN_clf(nn_classifier, X)
Y_pred_train = pr_train>0.5
plot(X[,1],X[,2],col=Y_pred_train+3,xlim = c(0,1), ylim = c(0,1))
accuracy_nn_train = sum(Y_pred_train==Y_clf)/length(Y_clf) #0.989

# prediction on testing set
pr_test = pred_NN_clf(nn_classifier, X_test)
Y_pred_test = pr_test>0.5
plot(X_test[,1],X_test[,2],col=Y_pred_test+3,xlim = c(0,1), ylim = c(0,1))
accuracy_nn_test = sum(Y_pred_test==Y_clf_test)/length(Y_clf_test) #0.985




################ Regression neural network ######################
relu <- function(x){
  x[ x[,1]<0, 1] = 0
  return(x)
}

NN_reg <- function(X_train, Y_train,  num_hidden = 100,
                  num_iterations = 2000, learning_rate = 1e-4){
  ## num_hidden: Number of hidden units in the hidden layer
  ## num_iterations: Number of iterations.
  ## learning_rate: Learning rate.
  n = dim(X_train)[1]
  p = dim(X_train)[2] + 1
  X_train1 = cbind(rep(1,n), X_train)
  
  alpha = matrix(rnorm(p*num_hidden)*0.01, nrow=p)
  beta = matrix(rnorm((num_hidden + 1))*0.01, nrow = num_hidden + 1)
  
  for (it in 1:num_iterations){
    # feed-forward
    Z = relu(X_train1 %*% alpha)
    Z1 = cbind(rep(1,n), Z)
    f_hat = Z1 %*% beta
    
    # error back propagation
    dbeta =  t(Y_train-f_hat) %*% Z1
    beta = beta +learning_rate * t(dbeta)
    
    
    for(k in 1:num_hidden)
    {
      da = (Y_train-f_hat) *beta[k+1] *( ( X_train1 %*% alpha[,k]) >=0)
      dalpha = t(da) %*% X_train1
      #dalpha = matrix(rep(1,n), nrow=1) %*% ((matrix(da, n, p)*X_train1))/n
      alpha[,k] = alpha[,k] + learning_rate *t(dalpha) #*n
      
    }
    print(it)
    print(sum(abs(dalpha)))
    print(sum(abs(dbeta)))
  }
  return( list(alpha = alpha, beta = beta) )

}

pred_NN_reg <- function(NN_regressor, X_test){
  alpha = NN_regressor$alpha
  beta = NN_regressor$beta
  n = dim(X_test)[1]
  
  X_test1 = cbind(rep(1,n), X_test)
  Z_test = relu(X_test1 %*% alpha)
  
  Z1_test = cbind(rep(1,n), Z_test)
  f_hat = Z1_test %*% beta
  return(f_hat)
}



############### Regression Example #################
# training data
set.seed(200)
n = 4000
X = matrix(runif(2*n), nrow = n)
Y_reg = X[,1]^2 + X[,2]^2 + 0.1*rnorm(n)
Y_reg = as.matrix(Y_reg)

# testing data
set.seed(150)
n_test = 4000
X_test = matrix(runif(2*n_test), nrow = n_test)
Y_reg_real = X_test[,1]^2 + X_test[,2]^2
Y_reg_test = X_test[,1]^2 + X_test[,2]^2 + 0.1*rnorm(n_test)
Y_reg_test = as.matrix(Y_reg_test)
# visualize testing data
library(rgl)
plot3d(X_test[,1], X_test[,2], Y_reg_real, col="black", size=2)

## fit the neural network regressor
nn_regressor = NN_reg(X, Y_reg)

# prediction on training set
y_pred_train = pred_NN_reg(nn_regressor, X)
plot3d(X[,1], X[,2], y_pred_train, col="blue", size=2)

# prediction on testing set
y_pred_test = pred_NN_reg(nn_regressor, X_test)
plot3d(X_test[,1], X_test[,2], y_pred_test, col="red", size=2)


## evaluatin RSS
RSS_train = sum((Y_reg-y_pred_train)^2)  # 80.65513 for training set
# 60.68554 in L2 boosting for 4000 obs. Committee has 20 one layer trees.

RSS = sum((Y_reg_test-y_pred_test)^2) # 83.287 for test set
# 61.67992 in L2 boosting for 4000 obs. Committee has 20 one layer trees.
# 55.89354 in recursive regression tree for 4000 obs 

RSS_constant = sum((Y_reg_test-mean(Y_reg_test))^2)  # 762.1422 for 4000 obs 
model = lm(Y_reg_test~X_test)
RSS_lm = sum(model$residuals^2) # 83.26175 for 4000 obs




