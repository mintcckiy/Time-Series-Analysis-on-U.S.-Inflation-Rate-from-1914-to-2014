# recursive forecast for holdout dataset 
recursive_forecast<-function(tsobject,tsobject_2,order_seq){
  forecast<-c()
  for(i in 1:length(tsobject_2)){
    model<-arima(tsobject,order=order_seq,include.mean=F,optim.control=list(maxit = 1000))
    forecast_point<-predict(model,1)
    forecast<-c(forecast,forecast_point$pred[1])
    tsobject<-c(tsobject,tsobject_2[i])
  }
  forecast 
}

# rolling window forecast for holdout dataset 
rolling_window<-function(tsobject,tsobject_2,order_seq){
  forecast<-c()
  for(i in 1:length(tsobject_2)){
    model<-arima(tsobject,order=order_seq,include.mean=F,optim.control=list(maxit = 10000))
    forecast_point<-predict(model,1)
    forecast<-c(forecast,forecast_point$pred[1])
    tsobject<-c(tsobject,tsobject_2[i])
    tsobject<-tsobject[2:length(tsobject)]
  }
  forecast 
}


# functions for computing the goodness-of-fit measure on the holdout dataset 

computeMSE <- function(y, pred){
  MSE = c()
  for (i in 1:length(y)){
    s <- ((y[i]-pred[i])^2)
    MSE <- c(MSE,s)
  }
  return(sum(MSE)/length(y))
}

computeMAD <- function(y, pred){
  MAD = c()
  for (i in 1:length(y)){
    s <- abs(y[i]-pred[i])
    MAD <- c(MAD,s)
  }
  return(sum(MAD)/length(y)) 
}

calculateMAPE<-function(holdout, preds){
  len<-length(holdout)
  total<-0 
  for(i in 1:len){
    total<-total+abs((holdout[i]-preds[i])/holdout[i])
    #print(total)
  }
  total/len
}

calculateAMAPE<-function(holdout, preds){
  len<-length(holdout)
  total<-0 
  for(i in 1:len){
    total<-total+abs((holdout[i]-preds[i])/(holdout[i]+preds[i]))
    #print(total)
  }
  total/len
}


# ArchLM test 
archlmtest <- function (x, lags, demean = FALSE) 
{
  x <- as.vector(x)
  if(demean) x <- scale(x, center = TRUE, scale = FALSE)
  lags <- lags + 1
  mat <- embed(x^2, lags)
  arch.lm <- summary(lm(mat[, 1] ~ mat[, -1]))
  STATISTIC <- arch.lm$r.squared * length(resid(arch.lm))
  names(STATISTIC) <- "Chisq"
  PARAMETER <- lags - 1
  names(PARAMETER) <- "df"
  PVAL <- 1 - pchisq(STATISTIC, df = PARAMETER)
  names(PVAL) <- "PVAL"
  METHOD <- "ARCH LM-test"
  result <- arch.lm
  df <- data.frame("Chisq" = STATISTIC, "Pval" = PVAL)
  return(df)
}

archlmbatch <- function (x, maxlags, demean = FALSE) 
{
  cat('Lag','\t','ChiSq','\t','PVal','\n')
  for (i in 1:maxlags)
  {
    temp <- archlmtest(x, i, demean)
    cat(i,'\t',temp$Chisq,'\t',temp$Pval,'\n')
  }
}


# recursive forecast for final ARMA-GARCH model 
arima_arch_garch_forecast_recursive<-function(Tsobject,formula,n_ahead){
  Forecast<-c()
  Upper_Bound<-c()
  Lower_Bound<-c()
  for(i in 1:n_ahead){
    model<-garchFit(formula,data=Tsobject)
    prediction<-predict(model,1)
    meanForecast<-prediction[1,1]
    upper<-1.96*prediction[1,3]+meanForecast
    lower<-meanForecast-1.96*prediction[1,3]
    Upper_Bound<-c(Upper_Bound,upper)
    Lower_Bound<-c(Lower_Bound,lower)
    Forecast<-c(Forecast,meanForecast)
    Tsobject<-c(Tsobject,meanForecast)
  }
  list(Forecast,Upper_Bound,Lower_Bound)
}

# rolling window forecast for final ARMA-GARCH model 
arima_arch_garch_forecast_rolling_window<-function(Tsobject,formula,n_ahead){
  Forecast<-c()
  Upper_Bound<-c()
  Lower_Bound<-c()
  for(i in 1:n_ahead){
    model<-garchFit(formula,data=Tsobject)
    prediction<-predict(model,1)
    meanForecast<-prediction[1,1]
    upper<-1.96*prediction[1,3]+meanForecast
    lower<-meanForecast-1.96*prediction[1,3]
    Upper_Bound<-c(Upper_Bound,upper)
    Lower_Bound<-c(Lower_Bound,lower)
    Forecast<-c(Forecast,meanForecast)
    Tsobject<-c(Tsobject,meanForecast)
    Tsobject<-Tsobject[2:length(Tsobject)]
  }
  list(Forecast,Upper_Bound,Lower_Bound)
}
