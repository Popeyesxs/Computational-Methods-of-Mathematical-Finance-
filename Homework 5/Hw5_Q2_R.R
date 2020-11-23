#question 2 Calibration of Heston Model
rm(list=ls())
getwd()
library(readxl)
opt_data <- read_excel("/Users/popeye/Desktop/MF 796/Homework 5/mf796-hw5-opt-data.xlsx",col_names=TRUE)
call<-opt_data[,1:3]
call['mid_price'] = (opt_data$call_bid+opt_data$call_ask)/2
call['spread'] = opt_data$call_ask-opt_data$call_bid
put<-opt_data[,1:3]
put['mid_price'] = (opt_data$put_bid+opt_data$put_ask)/2
put['spread'] = opt_data$put_ask-opt_data$put_bid 
#(a)

## Check the option prices for arbitrage
#Call (put) prices that are monotonically decreasing (increasing) in strike.

monotonicity = function(t, p){
  if (t == 'call'){
    return(all(p == cummin(p)))
  } else {
    return(all(p == cummax(p)))
  }
}

for (d in unique(call$expDays)){
  print(monotonicity('call', call[call$expDays == d,"mid_price"]))
  print(monotonicity('put', put[put$expDays == d,"mid_price"]))
}

# Rate of change relative to strike for call option prices is greater than -1and smaller than 0;
#Rate of change relative to strike for put option prices is greater than 0 and smaller than 1.

rate_change = function(t,p,k){
  if (t == 'call'){
    r = (c(p[2:length(p)],0)-p)/(c(k[2:length(p)],0)-k)
    r = r[1:(length(r)-1)]
    #print(r[1:(length(r)-1)])
    return(all(r>-1 & r<0))
  } else {
    r = (c(p[2:length(p)],0)-p)/(c(k[2:length(p)],0)-k)
    r = r[1:(length(r)-1)]
    #print(r[1:(length(r)-1)])
    return(all(r>0 & r<1))
  }
}

for (d in unique(call$expDays)){
  print(rate_change('call', call[call$expDays == d,"mid_price"]$mid_price,call[call$expDays == d,"K"]$K))
  print(rate_change('put', put[put$expDays == d,"mid_price"]$mid_price,put[df$expDays == d,"K"]$K))
}

#Both call and put prices are convex with respect to changes in strike

convexity = function(p){
  n = p-2*c(p[2:length(p)],0)+c(p[3:length(p)],0,0)
  return(all(n[1:(length(n)-2)]>0))
}

for (d in unique(call$expDays)){
  print(convexity(call[call$expDays == d,"mid_price"]$mid_price))
  print(convexity(put[put$expDays == d,"mid_price"]$mid_price))
}

#(b)

Heston_fft<-function( alpha, n, B, K, params ) {
  S0     = params[6]
  r      = params[7]
  T      = params[8]
  
  N = 2^n
  Eta = B / N
  Lambda_Eta = 2*pi/N
  Lambda = Lambda_Eta/Eta
  
  J = 1:N
  vj = (J-1)*Eta
  m = 1:N
  Beta = log(S0) - Lambda*N / 2
  km = Beta + (m-1)*Lambda
  
  ii <- complex(real=0, imaginary=1)
  #calculate values of characteristic function
  Psi_vj = rep(0, length(J))
  for (zz in 1:N) {
    u <- (vj[[zz]] - (alpha+1.0)*ii)
    numer <- Heston_cf(u, params) 
    denom <- ((alpha + ii* vj[[zz]]) * (alpha + 1.0 + ii*vj[[zz]]))
    
    Psi_vj[[zz]] <- (numer / denom)
  }
  
  #compute fft
  XX = (Eta/2)*Psi_vj*exp(-ii*Beta*vj)*(2-dirac(J-1))
  ZZ = fft(XX)
  
  #calculate option prices
  Multiplier = exp(-alpha*km)/pi
  ZZ2 <- Multiplier*ZZ
  Km <- exp(km)
  
  #discard strikes that are 0 or infinity to avoid errors in interpolation
  inds <- (!is.infinite(Km) & !is.nan(Km) & (Km > 1e-16) & (Km < 1e16))
  px_interp <- approx(Km[inds], Re(ZZ2[inds]), method = "linear", xout=K)
  
  fft_price = Re(exp(-r*T)*px_interp$y)
  return(fft_price)
}



#define a dirac delta function
dirac<-function(n) {
  y <- rep(0, length(n))
  y[n==0] = 1
  return(y)
}


#define a function that computes the characteristic function for variance gamma
Heston_cf<-function(u, params) {
  sigma <- params[1]
  eta0 <- params[2]
  kappa <- params[3]
  rho     <- params[4]
  theta     <- params[5]
  S0     <- params[6]
  r      = params[7]
  q      = params[8]
  T      = params[9]
  
  ii <- complex(real=0, imaginary=1)
  
  l = sqrt(sigma^2*(u^2+ii*u)+(kappa-ii*rho*sigma*u)^2)
  
  w = exp(ii*u*log(S0)+ii*u*(r-q)*T + 
            kappa*theta*T*(kappa-ii*rho*sigma*u)/sigma^2)/(cosh(l*T/2)+(kappa-ii*rho*sigma*u)/l*sinh(l*T/2))^(2*kappa*theta/sigma^2)
  
  y = w*exp(-(u^2+ii*u)*eta0/(l/tanh(l*T/2)+kappa-ii*rho*sigma*u))
  
  return(y)
}

### calibration 
library(optimx)

S0 = 267.15
r = 0.015
q = 0.0177

optim_data <- data.frame(S0=S0,
                         r=r,
                         q=q,
                         calls = call,
                         puts =put)

obj_fxn<-function(par,data) {
  kappa <- par[1]
  theta <- par[2]
  sigma <- par[3]
  rho   <- par[4]
  eta0  <- par[5]
  S0    <- data$S0[1]
  r     <- data$r[1]
  q     <- data$q[1]
  
  sse = 0
  # computing square errors using option data
  for (t in unique(data$calls.expT)){
    c = data[data$calls.expT == t,"calls.mid_price"] 
    p = data[data$puts.expT == t,"puts.mid_price"] 
    k = data[data$calls.expT == t,"calls.K"] 
    pars <- c(sigma,eta0,kappa,rho,theta,S0,r,q,t)
    alpha <- 1
    n <- 12
    B <- 1000.0
    sse <- sse+sum((Heston_fft(alpha, n, B, k, pars)-c)^2)
    # computing the put option price by choosing a negative alpha. We need to adjust alpha to say 1.25 or we'll get a division by zero errror.
    alpha <- -1.5
    sse <- sse+sum((Heston_fft(alpha, n, B, k, pars)-p)^2)
  }
  
  return(sse)
}

#(c)
guess <- c(0.2,0.1,0.5,0.1,0.5)
#guess <- c(0.25,0.06,0.8,-0.25,0.09)
#guess <- c(1.5,0.1,0.25,-0.5,0.1)

lower <- c(0.001,  0.0, 0.0, -1, 0.0)
upper <- c(5.0,  2.0, 2.0, 1, 1.0)
#lower <- c(0.001,0,0,-1,0)
#upper <- c(2.5,1,2,0.2,0.5)

opt_x <- optimx (guess ,
                 obj_fxn ,
                 gr = NULL ,
                 hess = NULL ,
                 lower = lower ,
                 upper = upper ,
                 method =c("Nelder-Mead", "BFGS"),
                 itnmax =NULL,
                 hessian =FALSE,
                 control = list(trace =1, abstol =1e-4),
                 optim_data)
opt_x

#(d) makes the weights inversely proportional to the quoted bid-ask spread
obj_fxn_w<-function(par,data) {
  kappa <- par[1]
  theta <- par[2]
  sigma <- par[3]
  rho   <- par[4]
  eta0  <- par[5]
  S0    <- data$S0[1]
  r     <- data$r[1]
  q     <- data$q[1]
  
  sse = 0
  # computing square errors using option data
  for (t in unique(data$calls.expT)){
    c = data[data$calls.expT == t,"calls.mid_price"] 
    p = data[data$puts.expT == t,"puts.mid_price"] 
    k = data[data$calls.expT == t,"calls.K"] 
    w_c = 1/data[data$calls.expT == t,"calls.spread"] 
    w_p = 1/data[data$puts.expT == t,"puts.spread"] 
    pars <- c(sigma,eta0,kappa,rho,theta,S0,r,q,t)
    alpha <- 1
    n <- 12
    B <- 1000.0
    sse <- sse+sum(w_c*(Heston_fft(alpha, n, B, k, pars )-c)^2)
    # computing the put option price by choosing a negative alpha. We need to adjust alpha to say 1.25 or we'll get a division by zero errror.
    alpha <- -1.5
    sse <- sse+sum(w_p*(Heston_fft(alpha, n, B, k, pars )-p)^2)
  }
  #print(sse)
  
  return(sse)
}

#guess <- c(0.05,0.2,0.2,-0.6,0.03)
guess <- c(0.25,0.06,0.8,-0.25,0.09)


lower <- c(0.001,  0.0, 0.0, -1, 0.0)
upper <- c(5.0,  2.0, 2.0, 1, 1.0)

#lower <- c(0.001,0,0,-1,0)
#upper <- c(2.5,1,2,0.2,0.5)



opt_x_w<- optimx (guess ,
                  obj_fxn_w ,
                  gr = NULL ,
                  hess = NULL ,
                  lower = lower ,
                  upper = upper ,
                  method =c("Nelder-Mead", "BFGS"),
                  itnmax =NULL,
                  hessian =FALSE,
                  control = list(trace =1, abstol =1e-4),
                  optim_data)















