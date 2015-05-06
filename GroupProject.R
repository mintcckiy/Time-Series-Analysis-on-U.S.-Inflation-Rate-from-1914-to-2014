library(fGarch)
library(forecast)
library(fUnitRoots)
library(tseries)
source("GroupProject_Functions.R")

# The dataset we chose is the U.S. Monthly Inflation Rate starting from Jan, 1914 to Sep, 2014. 
inflation <- read.csv('US Monthly Inflation Rate 1914-2014.csv',header=T)

# Format the raw date into workable time series data format 
inflation <- inflation[2:length(inflation)]
inflation <- ts(c(t(inflation)))
inflation <- inflation[!is.na(inflation)]

# Plot the time series 
plot(ts(inflation,start=c(1914,1),end=c(2014,9),frequency=12),type="l",
     main="Times Series plot of monthly U.S. Inflation Rate",xlab="Year",ylab="Inflation Rate")

# mystl <- stl(ts(inflation,start=c(1914,1),end=c(2014,9),frequency=12),s.window="periodic")
# mystl

# extract the reminder 
reminder<-mystl$time.series[,3]
acf(ts(reminder),lag=100)
adf.test(ts(reminder))
Box.test(ts(reminder),lag=20,type="Ljung-Box")
acf(ts(reminder),lag=400)
pacf(ts(reminder),lag=400)

# Plot the ACF and PACF of the time series 
acf(ts(inflation),lag=1000,main="Plot of Autocorrelation Function",xlab="Lag",ylab="ACF")
pacf(ts(inflation),lag=1000,main="Plot of Partial Autocorrelation Function",xlab="Lag",ylab="PACF")

# Tests for stationarity 
adf.test(ts(inflation))
pp.test(ts(inflation))
urersTest(ts(inflation),type="DF-GLS")
# Passed all the stationarity tests


# build the training set as the inflation_train and hold set as the inflation_test
inflation_train <- ts(inflation[1:floor(0.9*length(inflation))])
inflation_test <- ts(inflation[(floor(0.9*length(inflation))+1):length(inflation)])

# We use auto.arima() function to help us select the best model using different Information Criteria, AIC, BIC, and AICC 
# For each Information Criteria, we try both stepwise and brute force search 
# We go through both stationary models and non-stationary models 


# Stationary models 
# AIC 
sta_model_aic_s<-auto.arima(ts(inflation_train),ic="aic",seasonal=F,stationary=T,approx=F,stepwise=T,max.p=5, max.q=2,max.order=5)
sta_model_aic<-auto.arima(ts(inflation_train),ic="aic",seasonal=F,stationary=T,approx=F,stepwise=F,max.p=5, max.q=2,max.order=5)

# BIC
sta_model_bic_s<-auto.arima(ts(inflation_train),ic="bic",seasonal=F,stationary=T,approx=F,stepwise=T,max.p=5, max.q=2,max.order=5)
sta_model_bic<-auto.arima(ts(inflation_train),ic="bic",seasonal=F,stationary=T,approx=F,stepwise=F,max.p=5, max.q=2,max.order=5)

# AICC 
sta_model_aicc_s<-auto.arima(ts(inflation_train),ic="aicc",seasonal=F,stationary=T,approx=F,stepwise=T,max.p=5, max.q=2,max.order=5)
sta_model_aicc<-auto.arima(ts(inflation_train),ic="aicc",seasonal=F,stationary=T,approx=F,stepwise=F,max.p=5, max.q=2,max.order=5)

# Non-stationary models
# AIC 
non_model_aic_s<-auto.arima(ts(inflation_train),ic="aic",seasonal=F,stationary=F,approx=F,stepwise=T,max.p=5, max.q=2,max.order=5)
non_model_aic<-auto.arima(ts(inflation_train),ic="aic",seasonal=F,stationary=F,approx=F,stepwise=F,max.p=5, max.q=2,max.order=5)

# BIC
non_model_bic_s<-auto.arima(ts(inflation_train),ic="bic",seasonal=F,stationary=F,approx=F,stepwise=T,max.p=5, max.q=2,max.order=5)
non_model_bic<-auto.arima(ts(inflation_train),ic="bic",seasonal=F,stationary=F,approx=F,stepwise=F,max.p=5, max.q=2,max.order=5)

# AICC 
non_model_aicc_s<-auto.arima(ts(inflation_train),ic="aicc",seasonal=F,stationary=F,approx=F,stepwise=T,max.p=5, max.q=2,max.order=5)
non_model_aicc<-auto.arima(ts(inflation_train),ic="aicc",seasonal=F,stationary=F,approx=F,stepwise=F,max.p=5, max.q=2,max.order=5)

# The auto.arima() returns three models which it considers the best using different Information criteria and searching both stationary 
# and non-stationary space
# The models are ARIMA(4,1,0), ARIMA(1,1,1) and ARIMA(3,0,2)
# We will use the goodness-of-fit (MSE, MAD, MAPE and AMAPE) on the forecast of the holdout dataset to determine which is the our final model

# goodness-of-fit section 
# recursive forecast 
forecast_1_re<-recursive_forecast(inflation_train,inflation_test,c(4,1,0))
forecast_2_re<-recursive_forecast(inflation_train,inflation_test,c(3,0,2))
forecast_3_re<-recursive_forecast(inflation_train,inflation_test,c(1,1,1))

# rolling forecast 
forecast_1_ro<-rolling_window(inflation_train,inflation_test,c(4,1,0))
forecast_2_ro<-rolling_window(inflation_train,inflation_test,c(3,0,2))
forecast_3_ro<-rolling_window(inflation_train,inflation_test,c(1,1,1))

# compute the goodness-of-fit for model 1 c(4,1,0)
mse_re_1<-computeMSE(inflation_test,forecast_1_re)
mse_ro_1<-computeMSE(inflation_test,forecast_1_ro)
mad_re_1<-computeMAD(inflation_test,forecast_1_re)
mad_ro_1<-computeMAD(inflation_test,forecast_1_ro)
mape_re_1<-calculateMAPE(inflation_test,forecast_1_re)
mape_ro_1<-calculateMAPE(inflation_test,forecast_1_ro)
amape_re_1<-calculateAMAPE(inflation_test,forecast_1_re)
amape_ro_1<-calculateAMAPE(inflation_test,forecast_1_ro)

# compute the goodness-of-fit for model 2 c(3,0,2)
mse_re_2<-computeMSE(inflation_test,forecast_2_re)
mse_ro_2<-computeMSE(inflation_test,forecast_2_ro)
mad_re_2<-computeMAD(inflation_test,forecast_2_re)
mad_ro_2<-computeMAD(inflation_test,forecast_2_ro)
mape_re_2<-calculateMAPE(inflation_test,forecast_2_re)
mape_ro_2<-calculateMAPE(inflation_test,forecast_2_ro)
amape_re_2<-calculateAMAPE(inflation_test,forecast_2_re)
amape_ro_2<-calculateAMAPE(inflation_test,forecast_2_ro)

# compute the goodness-of-fit for model 3 c(1,1,1)
mse_re_3<-computeMSE(inflation_test,forecast_3_re)
mse_ro_3<-computeMSE(inflation_test,forecast_3_ro)
mad_re_3<-computeMAD(inflation_test,forecast_3_re)
mad_ro_3<-computeMAD(inflation_test,forecast_3_ro)
mape_re_3<-calculateMAPE(inflation_test,forecast_3_re)
mape_ro_3<-calculateMAPE(inflation_test,forecast_3_ro)
amape_re_3<-calculateAMAPE(inflation_test,forecast_3_re)
amape_ro_3<-calculateAMAPE(inflation_test,forecast_3_ro)

# The final model we chose is ARIMA(4,1,0)


# Sanity check for the the final model 
final_model<-arima(inflation,c(4,1,0))
resi<-final_model$residuals
sq_resi<-(final_model$residuals)^2
acf(resi,lag=1000,main="Plot of Autocorrelation Function of\nthe Residuals")
acf(sq_resi,lag=1000,main="Plot of Autocorrelation Function of\nthe Squared Residuals of ARIMA(4,1,0) Model")
pacf(resi,lag=1000)
Box.test(resi,type="Ljung-Box",lag=100)
# Didn't pass the Box-Ljung test but we still have to check the ARCH-GARCH effects 


# if there are arch/garch effect?
arch<-archlmbatch(inflation,400)
# There is a ARCH(1) effect 

# we cannot fit an ARIMA model to the garchFit so we difference it and fit an ARMA model instead 
diff<-diff(inflation)
myfit<- garchFit(formula = ~arma(4,0)+garch(1,0), data=diff)
myfit 

# sanity check for the ARMA-GARCH model to see if there is GARCH effect left in the model 
attributes(myfit)
residual_garch<-myfit@residuals
acf(residual_garch,lag=1000,main="Plot of Autocorrelation Function of Residuals\nof ARMA(4,0)-ARCH(1) Model")
pacf(residual_garch,lag=1000)
#s_res<-residuals(myfit, standardize = TRUE)
Box.test(s_res,type="Ljung-Box",lag=100)
# verified and it the residuals are not white noise


# predict the next five values using the final model and generate the 95% prediction interval 
formula <- ~arma(4,0)+garch(1,1)

# recursive forecast 
next_five_re<-arima_arch_garch_forecast_recursive(ts(diff),formula,5)
next_five_re

# rolling window forecast 
next_five_ro<-arima_arch_garch_forecast_rolling_window(ts(diff),formula,5)
next_five_ro

# plot the time series with forecast values and confidence interval 

# with recursive forecast 
plot(diff(ts(inflation,start=c(1914,1),end=c(2014,9),frequency=12)),main="Plot of Differenced Inflation Rate, Recursive \nForecast for Next Five Months \nand 95% Prediction Interval", xlab="Year",ylab="Difference of Inflation Rate",xlim=c(2000,2016))
x<-seq(2014.75,2015+1/12,1/12)
lines(x,y=next_five_re[[1]],col="red")
lines(x,y=next_five_re[[2]],col="blue")
lines(x,y=next_five_re[[3]],col="blue")

# with rolling window forecast 
plot(diff(ts(inflation,start=c(1914,1),end=c(2014,9),frequency=12)),main="Plot of Differenced Inflation Rate, Rolling Window \nForecast for Next Five Months \nand 95% Prediction Interval", xlab="Year",ylab="Difference of Inflation Rate",xlim=c(2000,2016))
x<-seq(2014.75,2015+1/12,1/12)
lines(x,y=next_five_ro[[1]],col="red")
lines(x,y=next_five_ro[[2]],col="blue")
lines(x,y=next_five_ro[[3]],col="blue")












