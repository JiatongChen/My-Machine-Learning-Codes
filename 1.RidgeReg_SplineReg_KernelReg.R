

#################################
## Function 1: Sweep operation ##
#################################

mySweep <- function(A, m){
  # Perform a SWEEP operation on A with the pivot element A[m,m].
  # A: a square matrix.
  # m: the pivot element is A[m, m].
  # Returns a swept matrix.
  n <- nrow(A)
  
  for(k in 1:m){ 
    for(i in 1:n)     
      for(j in 1:n)   
        if(i != k  & j != k)     
          A[i,j] <- A[i,j] - A[i,k]*A[k,j]/A[k,k]    
        
        for(i in 1:n) 
          if(i != k) 
            A[i,k] <- A[i,k]/A[k,k]  
          
          for(j in 1:n) 
            if(j != k) 
              A[k,j] <- A[k,j]/A[k,k]
            
            A[k,k] <- - 1/A[k,k]
  }
  
  return(A)
  
}


##################################
## Function 2: Ridge regression ##
##################################

myRidge <- function(X, Y, lambda){
  
  # Perform ridge regression of Y on X.
  # X: a matrix of explanatory variables.
  # Y: a vector of dependent variables. Y can also be a 
  # matrix, as long as the function works.
  # lambda: regularization parameter (lambda >= 0)
  # Returns beta, the ridge regression solution.
  
  n = dim(X)[1]
  p = dim(X)[2]
  Z = cbind(rep(1,n), X, Y)
  A = t(Z) %*% Z
  
  D = diag(rep(lambda,p+2))
  D[p+2,p+2] = 0
  D[1,1] = 0
  A = A+D
  S = mySweep(A,p+1)
  beta_ridge = S[1:(p+1), p+2]
  ## Function outputs the vector beta_ridge, the 
  ## solution to the ridge regression problem. Beta_ridge
  ## should have p + 1 elements.
  return(beta_ridge)
  
}





####################################################
## Function 3: Piecewise linear spline regression ##
####################################################


mySpline <- function(x, Y, lambda, p = 100){
  
  # Perform spline regression of Y on x.
  # 
  # x: An n x 1 vector or n x 1 matrix of explanatory variables.
  # You can assume that 0 <= x_i <= 1 for i=1,...,n.
  # Y: An n x 1 vector of dependent variables. Y can also be an 
  # n x 1 matrix, as long as the function works.
  # lambda: regularization parameter (lambda >= 0)
  # p: Number of cuts to make to the x-axis.
  n = length(x)
  X = matrix(x,nrow=n)
  for (k in (1:(p-1))/p)
    X = cbind(X, (x>k)*(x-k))
  beta_spline = myRidge(X, Y, lambda)
  y.hat = cbind(rep(1,n),X) %*% beta_spline
  
  ## Function should a list containing two elements:
  ## The first element of the list is the spline regression
  ## beta vector, which should be p + 1 dimensional (here, 
  ## p is the number of cuts we made to the x-axis).
  ## The second element is y.hat, the predicted Y values
  ## using the spline regression beta vector. This 
  ## can be a numeric vector or matrix.
  output <- list(beta_spline = beta_spline, predicted_y = y.hat)
  return(output)
  
  
}


####################################################
## Function 4: kernel regression ##
####################################################


myKernel <- function(x, Y, lambda, gamma){
  
  
  n = length(x)
  x_row = matrix(rep(x,n),n)
  x_col = t(x_row)
  
  K = exp(-gamma*(x_row-x_col)^2)
  D = diag(rep(lambda,n))
  KD = K+D
  c = solve(KD) %*% Y
  y.hat = K %*% c
  
  
  output <- list(c_kernel = c, predicted_y = y.hat)
  return(output)
  
  
}




########################################################
## examples ##
########################################################

lambda = 10
sigma = 0.1
npts=1000
x = runif(npts)
x = sort(x)
Y = x^2+rnorm(npts)*sigma

com = mySpline(x, Y, lambda)
beta = com[[1]]
yhat_spline = com[[2]]
plot(x, Y, ylim = c(-.2, 1.2), col ="red")
par(new = TRUE)
plot(x,yhat_spline, ylim=c(-.2, 1.2), type = "l", col = "green")


ker = myKernel(x, Y, 0.1, 10)
yhat_ker = ker[[2]]
plot(x, Y, ylim = c(-.2, 1.2), col ="red")
par(new = TRUE)
plot(x,yhat_ker, ylim=c(-.2, 1.2), type = "l", col = "black")

