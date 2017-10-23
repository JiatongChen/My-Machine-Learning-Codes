################## Regression Trees ####################
myRegTree <- function(X, Y, num_region, num_predictors){
  # for growing a single tree, num_predictors = dim(X)[2]. 
  # Or num_predictors <= dim(X)[2] in random forest.
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
      #if(n_more==0){
      #  #print(c_more)
      #  print(X_more)
      #}
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
  if( rss[i_split,j_split] == rss_0){
    return(list(leaf_value=mean(Y[,1]))) # it is the leaf node
  }else{
    # further split it into two trees
    X_less_split = matrix(X[X[,j_split]<=X[i_split,j_split],],ncol=p)
    X_more_split = matrix(X[X[,j_split]>X[i_split,j_split],],ncol=p)
    
    Y_less_split = matrix(Y[X[,j_split]<=X[i_split,j_split],1],ncol=1)
    Y_more_split = matrix(Y[X[,j_split]>X[i_split,j_split],1],ncol=1)
    
    less_tree = myRegTree(X_less_split, Y_less_split, num_region)
    more_tree = myRegTree(X_more_split, Y_more_split, num_region)
    return(list(split_var = j_split, threshold = X[i_split,j_split], lt = less_tree, mt = more_tree))
  }

}

predRegTree <- function(x, tree){
  #input x should be a 1*p vector
  if(is.null(tree$leaf_value) == FALSE){
    return(tree$leaf_value)   
  }else{
    j_split = tree$split_var 
    thres = tree$threshold
    if(x[j_split] <= thres){
      return(predRegTree(x,tree$lt))
    }else{
      return(predRegTree(x,tree$mt))
    }
  }
}

################## Regression Random Forest ##################
Random_Forest_Reg <- function(num_trees, num_predictors, num_region, X, Y){
  # 1 <= num_predictors <= dim(X)[2]
  Y = as.matrix(Y)
  n = dim(X)[1]
  p = dim(X)[2]
  reg_forest = list()
  
  # get bootstrape sample
  for (i in 1:num_trees){
    this_obs = sample(seq(1,n),n,replace = TRUE)
    X_new = matrix(X[this_obs,],n)
    Y_new = matrix(Y[this_obs,],n)
    # fit a tree model using current bootstrape sample, select num_predictors to split when fitting
    regtree_new= myRegTree(X_new, Y_new, num_region, num_predictors)
    #print(regtree_new)
    reg_forest =c(reg_forest,list(regtree_new))
  }
  return(list(reg_forest = reg_forest, num_trees = num_trees, num_predictors = num_predictors,
              num_region = num_region))
}

pred_Random_Forest_Reg<- function(random_forest_reg,X){
  n = dim(X)[1]
  num_trees = random_forest_reg$num_trees
  reg_forest = random_forest_reg$reg_forest
  Y_pred = mat.or.vec(n,num_trees)
  
  for (j in 1:num_trees){
    for (i in (1:n)){
      Y_pred[i,j] = predRegTree(X[i,], reg_forest[[j]])
    }
  }
  y_pred = rowMeans(Y_pred)
  return(y_pred)
}

############### regression tree example #################
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
regtree = myRegTree(X, Y_reg, 5,num_predictors=2) #print(regtree)

## calculate predicted values using testing data
y_pred = as.matrix(mat.or.vec(n_test,1))
for (i in (1:n_test)){
  y_pred[i,1] = predRegTree(X_test[i,], regtree)
}
library(rgl)
plot3d(X_test[,1], X_test[,2], y_pred, col="blue", size=2)

## evaluatin RSS
RSS = sum((Y_reg_test-y_pred)^2) # 55.89354 for 4000 obs 
RSS_constant = sum((Y_reg_test-mean(Y_reg_test))^2)  # 762.1422 for 4000 obs 
model = lm(Y_reg_test~X_test)
RSS_lm = sum(model$residuals^2) # 83.26175 for 4000 obs

########## random forest regression example ############

regrandforest = Random_Forest_Reg(20, 1, 5, X, Y_reg) 
# Grow 20 trees. For each tree, split on one predictor.

y_pred_rf = pred_Random_Forest_Reg(regrandforest,X_test)

library(rgl)
plot3d(X_test[,1], X_test[,2], y_pred_rf, col="red", size=2)

RSS_rf = sum((Y_reg_test-y_pred_rf)^2) # 44.7728 for 4000 obs 
RSS_constant = sum((Y_reg_test-mean(Y_reg_test))^2)  
model = lm(Y_reg_test~X_test)
RSS_lm = sum(model$residuals^2)