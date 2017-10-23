
####################################################
## Gaussian Process ##
####################################################


myGP <- function(x, Y, lambda, gamma){
  
  
  n = length(x)
  x_row = matrix(rep(x,n),n)
  x_col = t(x_row)
  
  K = exp(-gamma*(x_row-x_col)^2)
  D = diag(rep(lambda,n))
  KD = K+D
  c = solve(KD) %*% Y
  y.hat = K %*% c
  
  V=mat.or.vec(n,1)
  V = diag( K-K %*% solve(KD) %*% K)
  y.hat.ub = y.hat + 1.96*sqrt(V)
  y.hat.lb = y.hat - 1.96*sqrt(V)
  
  output <- list(c_kernel = c, predicted_y = y.hat, y_lb = y.hat.lb, y_ub = y.hat.ub)
  return(output)
  
  
}




############## examples ##############

lambda = 10
sigma = 0.1
npts=1000
x = runif(npts)
x = sort(x)
Y = x^2+rnorm(npts)*sigma



gp = myGP(x, Y, 0.1, 10)
yhat = gp[[2]]
yhat_lb = gp[[3]]
yhat_ub = gp[[4]]
plot(x, Y, ylim = c(-.2, 1.2), col ="red")
par(new = TRUE)
plot(x,yhat, ylim=c(-.2, 1.2), type = "l", col = "black", lwd = 3)
lines(x,yhat_ub,type="l",col="blue",lwd = 2)
lines(x,yhat_lb,type="l",col="blue",lwd = 2)

# find optimal gamma
num_gamma = 10
gamma_seq = seq(0.01,20,length.out = num_gamma)
loglik = mat.or.vec(num_gamma,1)

n = length(x)
x_row = matrix(rep(x,n),n)
x_col = t(x_row)

for (i in 1:num_gamma){
  K = exp(-gamma_seq[i]*(x_row-x_col)^2)
  D = diag(rep(lambda,n))
  KD = K+D
  print(K)
  loglik[i] = t(Y) %*% solve(KD) %*% Y + log(det(KD))
  print(i)
}

plot(gamma_seq,loglik, ylim=c(0,20), type = "l", col = "green", lwd = 3)
