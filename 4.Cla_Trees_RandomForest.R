################# Classification Trees ###################
count_categories <- function(Y){
  Y = as.matrix(Y)
  Y_unique = unique(Y)
  Y_cate = dim(Y_unique)[1]
  count = mat.or.vec(Y_cate,1)
  for (i in 1:Y_cate){
    count[i] = sum(Y==Y_unique[i])
  }
  return(count)
}

myClfTree <- function(X, Y, num_region, num_predictors){
  
  n = dim(X)[1]
  p = dim(X)[2]
  
  #P1 = sum(Y)/n 
  #P0 = 1-P1
  #entropy_0 = - P1*log2(P1) - P0*log2(P0) # the smaller, the purer
  
  entropy_0 = - sum( count_categories(Y)/n * log2(count_categories(Y)/n) )
  
  predictors = sort(sample(seq(1,p),num_predictors,replace = FALSE))
  
  # find out the best split
  entropy = matrix(rep(entropy_0,n*p),n,p) 
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
        #P1_less = sum(Y[X[,j]<=X[i,j],1])/n_less
        #P0_less = 1-P1_less
        #entropy_less = - P1_less*log2(P1_less) - P0_less*log2(P0_less)
        #P1_more = sum(Y[X[,j]>X[i,j],1])/n_more
        #P0_more = 1-P1_more
        #entropy_more = - P1_more*log2(P1_more) - P0_more*log2(P0_more)
        
        entropy_less =  -sum( count_categories(Y[X[,j]<=X[i,j],1])/n_less * 
                                log2(count_categories(Y[X[,j]<=X[i,j],1])/n_less) )
        entropy_more =  -sum( count_categories(Y[X[,j]>X[i,j],1])/n_more * 
                                log2(count_categories(Y[X[,j]>X[i,j],1])/n_more) )
        
        entropy[i,j] = n_less/n*entropy_less + n_more/n*entropy_more
        

        #print(sum(Y[X[,j]>X[i,j],1]))
        #print(n_more)
        #print(P1_more)
        #print(P0_more)
        #print(count_categories(Y[X[,j]>X[i,j],1]))
        #print(entropy_more)
        #print(entropy_less)
        #print(entropy[i,j])
        #print(entropy_0)
        
        if( entropy[i,j] >= entropy_0 ){
          entropy[i,j] = entropy_0
        }
      }
    }
  }
  
  index = arrayInd(which.min(entropy), dim(entropy))
  i_split = index[1,1]
  j_split = index[1,2]
  
  # decide if we want to split
  if( entropy[i_split,j_split] == entropy_0){
    return(list(leaf_value= mean(Y[,1])>0.5 )) # it is the leaf node
  }else{
    
    # further split it into two trees
    X_less_split = matrix(X[X[,j_split]<=X[i_split,j_split],],ncol=p)
    X_more_split = matrix(X[X[,j_split]>X[i_split,j_split],],ncol=p)
    
    Y_less_split = matrix(Y[X[,j_split]<=X[i_split,j_split],1],ncol=1)
    Y_more_split = matrix(Y[X[,j_split]>X[i_split,j_split],1],ncol=1)
    
    #n_less_split = dim(Y_less_split)[1]
    #n_more_split = dim(Y_more_split)[1]
    
    less_tree = myClfTree(X_less_split, Y_less_split, num_region)
    more_tree = myClfTree(X_more_split, Y_more_split, num_region)
    return(list(split_var = j_split, threshold = X[i_split,j_split], 
                lt = less_tree, mt = more_tree))
  }
}


predClfTree <- function(x, tree){
  #input x should be a 1*p matrix
  if(is.null(tree$leaf_value) == FALSE){
    return(tree$leaf_value)   
  }else{
    j_split = tree$split_var 
    thres = tree$threshold
    if(x[j_split] <= thres){
      return(predClfTree(x,tree$lt))
    }else{
      return(predClfTree(x,tree$mt))
    }
  }
}

################ Classification Random Forest #################
Random_Forest_Clf <- function(num_trees, num_predictors, num_region, X, Y){
  # 1 <= num_predictors <= dim(X)[2]
  Y = as.matrix(Y)
  n = dim(X)[1]
  p = dim(X)[2]
  
  clf_forest = list()
  # get bootstrape sample
  for (i in 1:num_trees){
    this_obs = sample(seq(1,n),n,replace = TRUE)
    X_new = matrix(X[this_obs,],n)
    Y_new = matrix(Y[this_obs,],n)
    # fit a tree model using current bootstrape sample,
    # select num_predictors to split when fitting
    clftree_new= myClfTree(X_new, Y_new, num_region, num_predictors)
    #print(regtree_new)
    clf_forest =c(clf_forest,list(clftree_new))
  }
  return(list(clf_forest = clf_forest, num_trees = num_trees, 
              num_predictors = num_predictors,num_region = num_region))
}

pred_Random_Forest_Clf <- function(random_forest_clf,X){
  n = dim(X)[1]
  num_trees = random_forest_clf$num_trees
  clf_forest = random_forest_clf$clf_forest
  Y_pred = mat.or.vec(n,num_trees)
  for (j in 1:num_trees){
    for (i in (1:n)){
      Y_pred[i,j] = predClfTree(X[i,], clf_forest[[j]])
    }
  }
  y_pred = rowMeans(Y_pred) > 0.5
  return(y_pred)
}



############## classification example ################
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

# visualize testing data
plot(X_test[,1],X_test[,2],col=Y_clf_test+1,xlim = c(0,1), ylim = c(0,1))

# fit the regression tree first
clftree = myClfTree(X, Y_clf, 5, 2) 

# calculate predicted values using testing data
y_pred = as.matrix(mat.or.vec(n_test,1))
for (i in (1:n_test)){
  y_pred[i,1] = predClfTree(X_test[i,], clftree)
}

plot(X_test[,1],X_test[,2],col=y_pred+1,xlim = c(0,1), ylim = c(0,1))


accuracy_clf = sum(y_pred==Y_clf_test)/length(Y_clf_test) # 0.961 for 1000 obs # 0.9375 for 400 obs

########## random forest regression example ############

clfrandforest = Random_Forest_Clf(20, 1, 5, X, Y_clf)
# Grow 20 trees. For each tree, split on one predictor.

y_pred_clf_rf = pred_Random_Forest_Clf(clfrandforest,X_test)

plot(X_test[,1],X_test[,2],col=y_pred_clf_rf+1,xlim = c(0,1), ylim = c(0,1))

accuracy_clf_rf = sum(y_pred_clf_rf==Y_clf_test)/length(Y_clf_test) # 0.971 for 1000 obs


