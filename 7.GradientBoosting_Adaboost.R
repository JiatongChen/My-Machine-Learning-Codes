################# one layer Classification Trees ###################
myClfTree <- function(X, Y, w, num_region=5){
  n = dim(X)[1]
  p = dim(X)[2]
  Y = as.matrix(Y)
  w = as.matrix(w)
  Y_tilde = w * Y
  
  H = matrix(mat.or.vec(n,1),n)
  H[,1] = 1
  alpha_0 = sum(Y_tilde[,1] * H[,1])/sum(H[,1] * H[,1]) 
  rss_0  = sum((Y_tilde[,1]-alpha_0*H[,1])^2)
  
  # find out the best split
  rss = matrix(rep(rss_0,n*p),n,p) 
  for (j in 1:p){
    for (i in 1:n){
      X_less = matrix(X[X[,j] <= X[i,j],],ncol=p)
      X_more = matrix(X[X[,j] > X[i,j],],ncol=p)
      n_less = dim(X_less)[1]
      n_more = dim(X_more)[1]
      
      if(n_less >= num_region & n_more >= num_region){
        H = matrix(mat.or.vec(n,1),n)
        H[X[,j]<=X[i,j],1] = 1
        H[X[,j]>X[i,j],1] = -1
        
        alpha = sum(Y_tilde[,1] * H[,1])/sum(H[,1] * H[,1]) 
        # find optimal alpha to minimize L2 loss
        rss[i,j]  = sum((Y_tilde[,1]-alpha*H[,1])^2)
        
        if( rss[i,j] >= rss_0 ){
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
    return(list(split = FALSE))# no splitting
  }else{
    #split it into two branches
    return(list(split_var = j_split, threshold = X[i_split,j_split], split = TRUE))
  }
}

predClfTree <- function(x, tree){
  #input x should be a 1*p vector
  #print(tree)
  if(tree$split == TRUE){
    j_split = tree$split_var 
    thres = tree$threshold
    
    if(x[j_split] <= thres){
      return(1)
    }else{
      return(-1)
    }
    
  }else{
    return(1)
  }
}


################ Gradient Boosting with Logistic Loss #################

GetWeight_GB <- function(Y,f_hat){
  w = 1/(1+exp(Y * f_hat))
  return(w)
}

GetVote_GB <- function( Y, f_hat, h, learning_rate = 0.001, num_iters = 1000){
  # f_hat: decision of current committee
  # h : decision of the new member
  beta = 1 # initial value
  for (it in 1:num_iters){
    grad = sum( (-Y*h) / (1+exp(Y*(f_hat+beta*h))) )
    beta = beta - learning_rate*grad
    #print(abs(grad))
    if( abs(grad) < 0.0001 ){
      break
    } 
  }
  return(beta)
}

predGraBoost <- function(X,committee){
  
  num_trees = length(committee)
  
  n = dim(X)[1]
  Y_pred = as.matrix(mat.or.vec(n,num_trees))
  votes = as.matrix(mat.or.vec(num_trees,1))
  
  for (j in 1:num_trees){

    for (i in 1:n){
      Y_pred[i,j] = predClfTree(X[i,], committee[[j]]$ClfTree)
    }
    votes[j,1] = committee[[j]]$Beta
  }
  y_pred = Y_pred %*% votes # need to put a sign function later on
  return(y_pred)
}


GraBoost <- function(X,Y, num_trees){
  
  n = dim(X)[1]
  p = dim(X)[2]
  committee = list()
  
  # initialize the first member
  h = matrix(rep(1,n),n)
  f_hat = matrix(rep(0,n),n)
  gamma = GetVote_GB(Y, f_hat,h)
  committee = c(committee, list(list(ClfTree=list(split = FALSE),Beta = gamma)))
  
  #print(length(committee))
  for (it in 2:num_trees){
    

    f_hat = predGraBoost(X,committee)

    w = GetWeight_GB(Y, f_hat)

    clftree = myClfTree(X, Y, w) 
    
    h = matrix(rep(0,n),n)
    for (i in 1:n){
      h[i,1] = predClfTree(X[i,], clftree)
    }

    beta =GetVote_GB( Y, f_hat,h)
    
    # What's the difference:
    # beta = GetVote_GB( Y, f_fat,h)
    # beta =GetVote_GB( Y, f_hat,h)

    committee = c(committee, list(list(ClfTree = clftree, Beta = beta)))
    
  }
  return(committee)
}

################ Adaboost #################
GetWeight_AdB <- function(Y,f_hat){
  w = exp(-Y * f_hat)
  D = w/sum(w)
  return(D)
}

GetVote_AdB <- function( Y, h, D){
  # h : decision of the new member
  a = sum(D[Y[,1]==h[,1],1])
  b = sum(D[Y[,1]!=h[,1],1])
  beta = 0.5*log(a/b)
  return(beta)
}

predAdaBoost <- function(X,committee){
  
  num_trees = length(committee)
  n = dim(X)[1]
  Y_pred = as.matrix(mat.or.vec(n,num_trees))
  votes = as.matrix(mat.or.vec(num_trees,1))
  
  for (j in 1:num_trees){
    for (i in 1:n){
      Y_pred[i,j] = predClfTree(X[i,], committee[[j]]$ClfTree)
    }
    #print(committee[[j]]$Beta)
    votes[j,1] = committee[[j]]$Beta
  }
  y_pred = Y_pred %*% votes # need to put a sign function later on
  return(y_pred)
}

AdaBoost <- function(X,Y, num_trees){
  
  n = dim(X)[1]
  p = dim(X)[2]
  committee = list()
  
  # initialize the first member
  f_hat = matrix(rep(0,n),n)
  D = GetWeight_AdB(Y, f_hat)
  clftree = myClfTree(X, Y, D)
  h = matrix(rep(0,n),n)
  for (i in 1:n){
    h[i,1] = predClfTree(X[i,], clftree)
  }
  
  gamma = GetVote_AdB(Y, h, D)
  committee = c(committee, list(list(ClfTree=clftree, Beta = gamma)))
  
  # start adding new members
  for (it in 2:num_trees){
    f_hat = predAdaBoost(X,committee)
    D = GetWeight_AdB(Y, f_hat)
    
    clftree = myClfTree(X, Y, D) 

    h = matrix(rep(0,n),n)
    for (i in 1:n){
      h[i,1] = predClfTree(X[i,], clftree)
    }
    
    beta =GetVote_AdB( Y, h, D)
    committee = c(committee, list(list(ClfTree = clftree, Beta = beta)))
  }
  return(committee)
}



####################### classification example ###########################
# training data
set.seed(200)
n = 1000
X = matrix(runif(2*n), nrow = n)
Y_clf = ((X[,1]^2 + X[,2]^2) <= 1)*2-1
Y_clf = as.matrix(Y_clf)

# testing data
set.seed(150)
n_test = 1000
X_test = matrix(runif(2*n_test), nrow = n_test)
Y_clf_test = ((X_test[,1]^2 + X_test[,2]^2) <= 1)*2-1
Y_clf_test = as.matrix(Y_clf_test)

# visualize testing data
plot(X_test[,1],X_test[,2],col=Y_clf_test+2,xlim = c(0,1), ylim = c(0,1))

############ fit the gradient boosting trees with logistic loss ##########
graboost_classifier = GraBoost(X,Y_clf, 100) # use 100 trees
# traing data
Y_pred = sign(predGraBoost(X,graboost_classifier))
plot(X[,1],X[,2],col=Y_pred+2,xlim = c(0,1), ylim = c(0,1))
accuracy_gb_train = sum(Y_pred==Y_clf)/length(Y_clf) # 0.998
# testing data
Y_pred_test = sign(predGraBoost(X_test,graboost_classifier))
plot(X_test[,1],X_test[,2],col=Y_pred_test+2,xlim = c(0,1), ylim = c(0,1))
accuracy_gb_test = sum(Y_pred_test==Y_clf_test)/length(Y_clf_test) # 0.976

########################## fit the adaboost ##############################
adaboost_classifier = AdaBoost(X,Y_clf, 100) # use 100 trees
# traing data
Y_pred_ada = sign(predAdaBoost(X,adaboost_classifier))
plot(X[,1],X[,2],col=Y_pred_ada+2,xlim = c(0,1), ylim = c(0,1))
accuracy_adb_train = sum(Y_pred_ada==Y_clf)/length(Y_clf) # 1
# testing data
Y_pred_ada_test = sign(predAdaBoost(X_test,adaboost_classifier))
plot(X_test[,1],X_test[,2],col=Y_pred_ada_test+2,xlim = c(0,1), ylim = c(0,1))
accuracy_ada_test = sum(Y_pred_ada_test==Y_clf_test)/length(Y_clf_test) #0.976



