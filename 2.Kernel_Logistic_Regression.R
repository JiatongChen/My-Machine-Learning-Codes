######### Kernel Logistic Regression #########
myKernel <- function(X,gamma){
  n = dim(X)[1]
  x1_row = matrix(rep(X[,1],n),n)
  x1_col = t(x1_row)
  
  x2_row = matrix(rep(X[,2],n),n)
  x2_col = t(x2_row)
  
  x1 = x1_row - x1_col
  x2 = x2_row - x2_col
  
  ker = exp(-gamma*(x1^2+x2^2))
  return(ker)
}

myKerLogistic <- function(X, Y, lambda, gamma, 
                num_iterations = 500, learning_rate = 1e-1){
  n <- dim(X)[1]
#  p <- dim(X)[2] + 1
#  X <- cbind(rep(1,n), X)
  
  ker = myKernel(X,gamma)
  coef <- matrix(rep(1,n), nrow = n)
  
  #acc <- rep(0, num_iterations)
  
  for (it in 1:num_iterations)
  {
    a = -colSums(  matrix(rep( ( 1/(1+exp((ker%*%coef)*Y)) )*Y,n) ,n) * ker )
    
    #a = -colSums(  (1/(1+exp( (ker%*%coef) %*% Y ))) * matrix(rep(Y,n),n) * ker )
    grad = matrix(a,n) + lambda* ker %*% coef
    #coef_old = coef
    coef = coef- grad*learning_rate
    
    print(norm(grad,"F") /n)
    if( norm(grad,"F")/n < 0.001 ){
      break
    } 
  }
  output <- list(kernel = ker, coef = coef)  
  return(output)
}
############### example #################
npts = 1000
X = matrix(runif(2*npts, -1,1),ncol = 2)
Y = ((X[,1]^2 + X[,2]^2) > 1)*(-2)+1
#plot(X[,1],X[,2],col=Y+2,xlim = c(-1,1), ylim = c(-1,1))

lambda = 0.0002
gamma =1.4
kerlog = myKerLogistic(X, Y, lambda, gamma,num_iterations = 1000,learning_rate = 0.001)
coef = kerlog$coef
ker = kerlog$kernel

ker %*% coef
Y_pred = sign(ker %*% coef)
plot(X[,1],X[,2],col=Y_pred+2,xlim = c(-1,1), ylim = c(-1,1))