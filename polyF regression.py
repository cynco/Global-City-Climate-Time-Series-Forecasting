# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.tools.plotting import autocorrelation_plot
from fbprophet import Prophet

# TODO: linear regression for future predictions
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#file = "../input/GlobalTemperatures.csv"
#file = "../input/GlobalLandTemperaturesByCity.csv"
file = "../input/GlobalLandTemperaturesByCountry.csv"
out = pd.read_csv(file, parse_dates=True, index_col=0)

#, parse_dates=True, index_col=0
#print(out.index)
print(out.describe())

# filter all data to Australia only and only 100 years
# changing this to 108 years
out = out.loc[out['Country'] == 'Australia']
#out = out.loc[out['City'] == 'Sydney']
out = out.loc[out.index.year > 1909]
out = out.loc[out.index.year <= 2012] # data only goes up to 2013? which isnt complete anyway

# if testing the global temperatures use this without Australia flag
# fields: LandAndOceanAverageTemperature, LandMinTemperature, LandMaxTemperature, LandAverageTemperature
#out['AverageTemperature'] = out['LandAndOceanAverageTemperature'];

# optional data cleaning: replace empty null values with X
# drop unnecessary columns
doClean = True
if doClean:
    
    # is there any null or empty data - 5 for AverageTemperature & AverageTemperatureUncertainty
    print('Null Values:\n', out.isnull().sum())
    
    # fill empty data columns with median - no! this is wrong for for older dates
    # need to either use previous, or average of X rows
    #out.loc[:,"AverageTemperature"] = out["AverageTemperature"].fillna(out["AverageTemperature"].median())
    #out.loc[:,"AverageTemperatureUncertainty"] = out["AverageTemperatureUncertainty"].fillna(out["AverageTemperatureUncertainty"].median())
    
    # alternative use bfill (back fill) or ffill (forward)
    out = out.ffill()
    
    # check data fill - should all be 0
    print('Null Values fixed:\n', out.isnull().sum())
    
    
    # drop unnecessary columns: country, uncertainty
    out = out.drop('AverageTemperatureUncertainty', 1)
    out = out.drop('Country', 1)
    
    # add year column instead of dt
    #out['dt'] = pd.to_datetime(out['dt'])
    #out['year'] = out['dt'].map(lambda x: x.year)
    #out['Year'] = out.index.year
    out['Year'] = out.index.year + ((out.index.month-1)/12) # convert year to decimal
    #print(out['Year'])
    out['Month'] = out.index.month
    out['Days delta'] = (out.index -  out.index[0]).days
    #0=summer,1=autumn,2=winter,3=spring
    out['Season'] = [1 if month >= 3 and month <= 5 else 2 if month >= 6 and month <= 8 else 3 if month >= 9 and month <= 11 else 0 for month in out['Month']]
    
    # what about adding a seasonal delta? eg subtract the seasonal average from the yearly average?
    #out_yearly_mean = out.resample('A').mean()
    ma = out.rolling(window=12).mean()
    out['MA'] = ma['AverageTemperature']
    out['Season offset'] = out['AverageTemperature'] - ma['AverageTemperature']
    
    # optional? dont need month anymore
    #out = out.drop('Month', 1)
    
    


# print info: count 1930, mean, 21.6,
#print(out)
print(out['AverageTemperature'].describe())
print(out['AverageTemperature'].shape)
out['AverageTemperature'].shape
#print(out['AverageTemperature'].value_counts())
#print(out['AverageTemperature'].value_counts(normalize = True))

# save subset
out.to_csv('AustraliaLandTemperatures.csv')

# resampling: http://benalexkeen.com/resampling-time-series-data-with-pandas/
# reduce the rows - by averaging fo rthe graph
out_years = out.resample('A').mean()
#out_quarters = out.resample('Q').mean()
#out_quarters = out.resample('Q', how=['mean', np.min, np.max])
#out_years = out.resample('A', how=['mean', 'min', 'max'])
#out_years_10 = out.resample('10A', how=['mean', 'min', 'max'])
out_years_10 = out.resample('10A').mean()


doSeasonal = False
if doSeasonal:
    # seasonal trends: http://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/ 
    seasonal_data = out[:24] # 12 months * 2 years
    sx = seasonal_data.index
    #out["Year"].values
    #.reshape(-1,1)
    #print("sx: ", sx)
    sy = seasonal_data['AverageTemperature'].values
    #print("sy: ", sy)
    seasonal_series = pd.Series(sy, index=sx)
    print("sd: ", seasonal_series)
    print("sd: ", seasonal_series.describe())
    seasonal_result = seasonal_decompose(seasonal_series, model='additive')
    #, freq=1)
    f = plt.figure()
    f = seasonal_result.plot()
    #plt.plot(seasonal_result)
    f.savefig("test-seasonal-plot.png")
    print("Seasonal resid: " , seasonal_result.resid)
    print("Seasonal s: " , seasonal_result.seasonal)
    print("Seasonal trend: " , seasonal_result.trend)

# rolling average is no different to the resample mean
#out["Rolling"] = pd.rolling_mean(out['AverageTemperature'],window=12)
#print(out["Rolling"])

# which data do we want to use: mean, min or max?
# std is highest for min temperatures (amin) 0.690336, 0.799962
#print(out_years)
print("SD: ", out_years["AverageTemperature"].std())
print("SD: ", out_years_10["AverageTemperature"].std())

#plot as graph
#plt.plot(out_quarters.index, out_quarters['AverageTemperature'],'-',markersize=1)
#plt.plot(out_years.index, out_years['AverageTemperature'],'-',markersize=1)
#plt.plot(out_years_10.index, out_years_10['AverageTemperature'],'-',markersize=1)
#plt.plot(out.index, out['Rolling'],'-',markersize=1)
#axes = plt.gca()
#axes.set_ylim([15,25])
#plt.savefig("aust-plot.png")


# save yearly summary
#out_years.to_csv('AustraliaLandTemperatures-year.csv')


    
    
# test a model + draw plot. 
# testing with Linear regression, SVR, and polynomial features with Ridge
# predictions are too linear
def testModel(data_out, labelX, labelY, filename, appendToLastPlot=False, doBay=False, doSvr=True, doPoly=True):
    
    # future predictions
    # https://www.kaggle.com/saksham219/temperature-variation-over-the-years-in-new-delhi
    #data_out = out.resample('A').mean()
    #data_out = data_out.reset_index()
    x = data_out["Year"].values.reshape(-1,1)
    #x = data_out.index.values.reshape(-1,1)
    #print(x) 
    #obtaining values of temperature
    y = data_out['AverageTemperature'].values
    #print(y)
    
    # Using linear regression and finding accuracy of our prediction
    # http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
    # http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge_variance.html
    #reg = Ridge()
    reg = LinearRegression()
    #reg = BayesianRidge()
    reg.fit(x,y)
    y_preds = reg.predict(x)
    # accuracy/variance R-squared valu (1 is perfect) # 0.294882742867
    print("Accuracy: ", reg.score(x,y))
    # The coefficients
    print('Coefficients: ', reg.coef_)
    # The mean squared error
    print("Mean squared error: ", np.mean((reg.predict(x) - y) ** 2))
    
    

    
    #plotting data along with regression
    if not appendToLastPlot:
        plt.figure()
    plt.plot(x, y, '-', markersize=1, label="raw")
    plt.plot(x, y_preds,'-', markersize=1, label="LR least squares")
    #plt.scatter(x=x, y=y_preds)
    #plt.scatter(x=x,y=y, c='r')
    
     # test predictions past/future of temperature
    print("Prediction 1917: ", reg.predict(1917))
    print("Prediction 2017: ", reg.predict(2017))
    print("Prediction 2117: ", reg.predict(2117))
    #2020: 22.34
    


    
    # BayesianRidge
    #doBay = False
    if doBay:
        b = BayesianRidge(alpha_1=1.0, lambda_1=1.0, alpha_2=1.0, lambda_2=1.0)
        b.fit(x,y)
        y_preds_b = b.predict(x)
        plt.plot(x, y_preds_b, '-', markersize=1, label="bayridge")
        
    # Using SVM for prediction
    #doSvr = True
    if doSvr:
        #svr = SVR(kernel='poly', C=1e3, degree=2) #too slow
        #svr = SVR(kernel='rbf')#
        #svr = SVR(C=1000, epsilon=0.0001)
        svr = SVR(kernel = "rbf", C = 1e3, gamma = 0.1, epsilon = 0.1)
        #svr = SVR(kernel='poly', degree=2)#(kernel='linear') 'rbf'
        svr.fit(x,y)
        y_preds_svr = svr.predict(x)
        plt.plot(x, y_preds_svr,'-', markersize=1, label="SVR")
        
        print("SVR Accuracy: ", svr.score(x,y))
        print("SVR Prediction 1917: ", svr.predict(1917))
        print("SVR Prediction 2017: ", svr.predict(2017.0))
        print("SVR Prediction 2017b: ", svr.predict(2017.5))
        print("SVR Prediction 2018: ", svr.predict(2018.0))
        print("SVR Prediction 2018b: ", svr.predict(2018.5))
        print("SVR Prediction 2117: ", svr.predict(2117))
    

    # http://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/
    # visualize the relationship between the features and the response using scatterplots
    #plt2 = sns.pairplot(data_out, x_vars=['Year'], y_vars='AverageTemperature', size=8, kind='reg')
    #plt2.savefig("output.png")
    
   
    
    # compare a prediction with an actual value
    #print("Actual 1967: ", data_out.loc[data_out['Year'] == 1967]['AverageTemperature'])
    

    # Using polynomial features with Ridge regression
    # in other words: instead of our straight least squares line from linear regression, apply the data to a curve
    # http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#sphx-glr-auto-examples-linear-model-plot-polynomial-interpolation-py
    #x_plot = data_out["Year"].values.reshape(-1,1)
    #doPoly = True
    if doPoly:
        model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0))
        model.fit(x, y)
        y_preds2 = model.predict(x)
        plt.plot(x, y_preds2, '-', markersize=1, label="Poly F (3) + Ridge")
        # todo: need to score the poly/ridge model
        print("Accuracy: ", model.score(x,y))
        
        # test predicitons - terrible with 6 features
        print("Prediction 1917: ", model.predict(1917))
        print("Prediction 2017: ", model.predict(2017))
        print("Prediction 2117: ", model.predict(2117))
        
        # winter vs summer prediction? incorrect!
        print("Prediction 2017 summer: ", model.predict(2017.0))
        print("Prediction 2017 winter: ", model.predict(2017.5))
        
    
    # image must be saved last
    if not appendToLastPlot:
        plt.ylabel(labelY)#('Temperature')
        plt.xlabel(labelX)#('Year')
        plt.legend(loc='upper left')
    #plt.ylim([10,25])
    #plt.savefig("future.png")
    plt.savefig(filename)
    #"future.png")
    print(filename)
    
    
# run the tests / save the plots
# raw, year mean, 10 year mean, min, max
testModel(out, "Year", "Raw temperature", "raw.png")
testModel(out.resample('A').mean(), "Year", "Mean temperature", "mean.png")
testModel(out.resample('10A').mean(), "Year", "10 year mean temperature", "10-mean.png")
testModel(out.resample('A').min(), "Year", "Min temperature", "min.png")
testModel(out.resample('A').max(), "Year", "Max temperature", "max.png")

# seasonal tests
# TODO: check the seasonal inverse - eg. lowest temperatures for summer, highest temperatures for winter
testModel(out.loc[out['Season'] == 0].resample('A').mean(), "Year", "Summer Mean temperature", "seasons-summer.png")
testModel(out.loc[out['Season'] == 2].resample('A').mean(), "Year", "Winter Mean temperature", "seasons-winter.png")
testModel(out.loc[out['Season'] == 1].resample('A').mean(), "Year", "Autumn Mean temperature", "seasons-autumn.png")
testModel(out.loc[out['Season'] == 3].resample('A').mean(), "Year", "Spring Mean temperature", "seasons-spring.png")

# combined seasonal tests - eg. show the yearly average and the winter average to compare
#def testModel(data_out, labelX, labelY, filename, appendToLastPlot=False, doBay=False, doSvr=True, doPoly=True):
plt.figure()
testModel(out.resample('A').mean(), "Year", "Mean temperature + seasonal", "mean-combined.png", True, False, False, False)
testModel(out.loc[out['Season'] == 2].resample('A').mean(), "Year", "Mean temperature + winter seasonal", "mean-combined-seasons-winter.png", True, False, False, False)
testModel(out.loc[out['Season'] == 0].resample('A').mean(), "Year", "Mean temperature + summer seasonal", "mean-combined-seasons-summer.png", True, False, False, False)


# autocorrelation
# http://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
plt.figure()
data_ = out.resample('A').mean()
ts = pd.Series(data_['AverageTemperature'].values, index=data_.index)
autocorrelation_plot(ts)
plt.savefig("autocorrelation_plot.png")

# seasonal offset - seasonal variation against 12 month rolling mean
# this is probably the most important graph- frequency trends towards erratic
plt.figure()
data_ = out.resample('5A').mean()
#data_ = out #.loc[out['Season'] == 2]
#.resample('5A').mean() # summer
#data_ = out.resample('A').mean()
plt.ylim([-.5, .5]) 
plt.plot(data_['Year'], data_['Season offset'], '-', markersize=1, label="season offset")
plt.ylabel('Seasonal offset')
plt.savefig("season-offsets.png")


# detrend - remove the average - for a +- delta in degrees
#data_ = out.resample('100A')
#data_avg = out.resample('A').mean()
#data_avg = data_avg.resample(rule='100A', fill_method='ffill')
#data_['AverageTemperature'] = data_['AverageTemperature'] - data_avg['AverageTemperature']
#testModel(data_ - data_avg, "Year", "Delta temperature", "delta.png")

# some weird outlier in spring data late 2000s? 2013 average was 17.748
# because there is only 1 month of data for spring - maybe drop 2013?
#print(data_spring_mean.loc[data_spring_mean['Year'] >= 2010])



# other model tests...


# ARIMA- auto regression intergrated moving average
doArima = False
if doArima:
    data_ = out.resample('A').mean()
    arima = ARIMA(data_['AverageTemperature'], order=(2,1,0))
    arima_fit = arima.fit()
    plt.figure()
    plt.plot(arima_fit.fittedvalues)
    #plt.plot(data_['AverageTemperature'])
    plt.savefig("arima-plot.png")



    
# Using facebook's prophet forecasting model
# https://facebookincubator.github.io/prophet/docs/quick_start.html
# https://www.kaggle.com/chinmaymk/predicting-landaveragetemperature-using-arima
def testProphet(data_, yearly_seasonality=True, periods=1200, freq='M', filename='AustraliaLandTemperatures-forecast', include_history=True):
    #data_ = out # all (monthly)
    #data_ = out.resample('Q').mean() # quaterly
    #data_ = out.resample('Q-NOV').mean() # quaterly by season - year ends in november
    # https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    #data_ = out.resample('A').mean() # yearly averages
    #data_ = out.loc[out['Month'] == 12] # december
    #data_ = out.loc[out['Season'] == 2] # winter
    
    # convert column names to make compatible with prophet
    subset = data_.reset_index()[['dt', 'AverageTemperature']]
    subset.rename(columns={"dt": "ds", "AverageTemperature": "y"}, inplace=True)

    # default params
    #pmodel = Prophet()
        #prophet(df = df, growth = "linear", changepoints = NULL,
#   n.changepoints = 25, yearly.seasonality = "auto",
#   weekly.seasonality = "auto", holidays = NULL,
#   seasonality.prior.scale = 10, holidays.prior.scale = 10,
#   changepoint.prior.scale = 0.05, mcmc.samples = 0, interval.width = 0.8,
#   uncertainty.samples = 1000, fit = TRUE, ...)
    
    # test scaling down seasonality from 10 to 1, 10 to 20. turn off yearly and weekly seasonality
    # seasonality_prior_scale does nothing when setting seasonality to false # seasonality_prior_scale=1, 
    # yearly_seasonality should be off when using yealry mean input
    pmodel = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=False)
    pmodel.fit(subset)
    
    # make a (100 year, 1 year, X year) forecast with (monthly M, yearly A) frequency
    #future = pmodel.make_future_dataframe(periods=12*100, freq='M')
    #future = pmodel.make_future_dataframe(periods=100, freq='A')
    #future = pmodel.make_future_dataframe(periods=4*20, freq='Q-NOV')
    future = pmodel.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
    forecast = pmodel.predict(future)
    print(forecast.tail())
    
    # plot the trend components - only useful if plotting ALL data, not subsets - eg. by season
    # switch this off for yearly mean, or not using yearly_seasonality
    if pmodel.yearly_seasonality:
        plt.figure()
        #plt.ylim([-15, 15]) 
        pmodel.plot_components(forecast);
        plt.savefig(filename + "-prophet-plot-components.png")
    
    # plot the forecast 
    plt.figure()
    pmodel.plot(forecast)
    #plt.plot(forecast['ds'], forecast['yhat'], '-')
    #plt.plot(forecast['ds'], forecast['trend'], '-')
    plt.ylabel('Temperature') #FB prophet')
    plt.xlabel('Year') #FB prophet')
    plt.suptitle(filename) #'Forecast')
    plt.savefig(filename + "-prophet-plot.png")
    
    # add another column to the data with the actual temps for years that we have from original data
    # note for forecast - the values will be empty 
    # https://stackoverflow.com/questions/27126511/add-columns-different-length-pandas
    forecast = pd.concat([forecast, subset['y']], axis=1)
    forecast.rename(columns={'y': 'AverageTemperature'}, inplace=True)
    
    # save results to csv
    forecast.to_csv(filename + '.csv')
    

# monthly raw values - 108 years
testProphet(out, True, 12*108, 'M', 'AustraliaLandTemperatures-forecast-monthly')

# yearly mean
testProphet(out.resample('A').mean(), False, 108, 'A', 'AustraliaLandTemperatures-forecast-yearly')

# tests: only historical data + 1 year future
#testProphet(out, True, 12*108, 'M', 'AustraliaLandTemperatures-forecast-monthly-historical', False)

#1912 v 2012 want to see the seasonal trend only first 5 years vs last 5 years
#testProphet(out.head(240), True, 12*1, 'M', 'AustraliaLandTemperatures-forecast-monthly-A')
#testProphet(out.tail(240), True, 12*1, 'M', 'AustraliaLandTemperatures-forecast-monthly-B')
