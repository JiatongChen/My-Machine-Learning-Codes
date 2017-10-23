################## one layer Regression Trees ####################
myRegTree <- function(X, Y, num_region, num_predictors){
  # for growing a single tree, num_predictors = dim(X)[2]. 
  n = dim(X)[1]
  p = dim(X)[2]
  rss_0 = sum((Y[,1]-mean(Y[,1]))^2)
  predictors = sort(sample(seq(1,p),num_predictors,replace = FALSE))
  
  # find out the best split
  rss = matrix(rep(rss_0,n*p),n,p) 
  for (j in predictors){
    for (i in 1:n){
      X_less = matrix(X[X[,j] <= X[i,j],],ncol=p)
      X_more = matrix(X[X[,j] > X[i,j],],ncol=p)
      
      c_less = matrix(mean(Y[X[,j] <= X[i,j],1]),ncol=1)
      c_more = matrix(mean(Y[X[,j] > X[i,j],1]),ncol=1)
      
      n_less = dim(X_less)[1]
      n_more = dim(X_more)[1]

      if(n_less >= num_region & n_more >= num_region){
        rss[i,j] = sum((Y[X[,j]<=X[i,j],1] - c_less)^2) + sum((Y[X[,j]>X[i,j],1] - c_more)^2)
        if(rss[i,j]>=rss_0){
          rss[i,j] = rss_0
        }
      }
    }
  }
  
  index = arrayInd(which.min(rss), dim(rss))
  i_split = index[1,1]
  j_split = index[1,2]
  
  # decide if we want to split
  #if( rss[i_split,j_split] == rss_0){
  #  return(list(leaf_value=mean(Y[,1]))) # it is the leaf node
  #}else{
   
   # further split it into two trees
    X_less_split = matrix(X[X[,j_split]<=X[i_split,j_split],],ncol=p)
    X_more_split = matrix(X[X[,j_split]>X[i_split,j_split],],ncol=p)
    
    Y_less_split = matrix(Y[X[,j_split]<=X[i_split,j_split],1],ncol=1)
    Y_more_split = matrix(Y[X[,j_split]>X[i_split,j_split],1],ncol=1)

    return(list(split_var = j_split, threshold = X[i_split,j_split], 
                lt = mean(Y_less_split[,1]) , mt =mean(Y_more_split[,1]) ))
  #}
  
}

predRegTree <- function(x, tree){
  #input x should be a 1*p vector
  j_split = tree$split_var 
  thres = tree$threshold
  if(x[j_split] <= thres){
    return(tree$lt)
  }else{
    return(tree$mt)
  }
}


############### L2 Boosting ################
L2_Boosting <- function(X,Y,num_trees){

  n = dim(X)[1]
  p = dim(X)[2]
  R = as.matrix(Y)
  committee = list()
  for (i in 1:num_trees){
    
    regtree = myRegTree(X, R, 5, p)
    committee = c(committee,list(regtree))
    
    y_pred = as.matrix(mat.or.vec(n,1))
    for (j in 1:n){
      y_pred[j,1] = predRegTree(X[j,], regtree)
    }
    R = R-y_pred
  }
  return(list(committee = committee, num_trees = num_trees))
  
}


pred_L2_Boosting <- function(X,L2_Boosting_regressor){
  
  committee = L2_Boosting_regressor$committee
  num_trees = L2_Boosting_regressor$num_trees
  n = dim(X)[1]
  Y_pred = mat.or.vec(n,num_trees)

  for (j in 1:num_trees){
    for (i in 1:n){
      Y_pred[i,j] = predRegTree(X[i,], committee[[j]])
    }
  }
  y_pred = rowSums(Y_pred)
  return(y_pred)
  
}


############### L2 boosting tree example #################
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


## fit the regression tree first
#regtree = myRegTree(X, Y_reg, 5,num_predictors=2) #print(regtree)

## fit the L2 boosting tree regressor
l2_boosting_regressor = L2_Boosting(X,Y_reg,50)

## calculate predicted values using training data
y_pred = pred_L2_Boosting(X,l2_boosting_regressor)
plot3d(X[,1], X[,2], y_pred, col="blue", size=2)

## calculate predicted values using testing data
y_pred_test = pred_L2_Boosting(X_test,l2_boosting_regressor)
plot3d(X_test[,1], X_test[,2], y_pred_test, col="red", size=2)

## evaluatin RSS
RSS_train = sum((Y_reg-y_pred)^2)  
# 60.68554 in L2 boosting for 4000 obs. Committee has 20 one layer trees.

RSS = sum((Y_reg_test-y_pred_test)^2) 
# 61.67992 in L2 boosting for 4000 obs. Committee has 20 one layer trees.
# 55.89354 in recursive regression tree for 4000 obs 

RSS_constant = sum((Y_reg_test-mean(Y_reg_test))^2)  # 762.1422 for 4000 obs 
model = lm(Y_reg_test~X_test)
RSS_lm = sum(model$residuals^2) # 83.26175 for 4000 obs

