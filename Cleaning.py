# FOLLOWING LIGHT GUIDANCE FROM : https://www.kaggle.com/code/nibukdk93/weather-data-analysis
# DATA FROM SAME SOURCE

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder # needed to install scikit learn, not jsut sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import copy

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/' \
       'EXTERNAL DATA SCIENCE PROJECTS 2023/Brazil Weather/Data Sets/'



# High level Time Series Data
weather_data = pd.read_csv(path+'sudeste.csv')

weather_data.describe() # over 97 million observations!
# count  9.779168e+06  9.779168e+06  ...  9.779168e+06  9.462694e+06
# mean   3.592531e+02  5.940923e+02  ...  1.385991e+02  4.494015e+00
# std    3.901630e+01  3.980379e+02  ...  1.052018e+02  2.981790e+00
# min    1.780000e+02  0.000000e+00  ...  0.000000e+00  0.000000e+00
# 25%    3.280000e+02  2.830000e+02  ...  5.600000e+01  2.300000e+00
# 50%    3.580000e+02  5.730000e+02  ...  1.140000e+02  4.200000e+00
# 75%    3.940000e+02  8.750000e+02  ...  2.160000e+02  6.300000e+00
# max    4.230000e+02  1.758000e+03  ...  3.600000e+02  5.000000e+01
# [8 rows x 25 columns]

print(weather_data.head(5))
print(weather_data.shape)
#(9779168, 31)


################################
# BRIEF EDA
###############################

# Visuals and stuff
# histograms ez


# Name of all columns
print(weather_data.columns)
print(weather_data.dtypes)
#['wsid', 'wsnm', 'elvt', 'lat', 'lon', 'inme', 'city', 'prov', 'mdct',
  #     'date', 'yr', 'mo', 'da', 'hr', 'prcp', 'stp', 'smax', 'smin', 'gbrd',
 #      'temp', 'dewp', 'tmax', 'dmax', 'tmin', 'dmin', 'hmdy', 'hmax', 'hmin',
#       'wdsp', 'wdct', 'gust']

numbers_only = weather_data.select_dtypes(include=['float64', 'int64'])
print(numbers_only.columns)
#['wsid', 'elvt', 'lat', 'lon', 'yr', 'mo', 'da', 'hr', 'prcp', 'stp',
#       'smax', 'smin', 'gbrd', 'temp', 'dewp', 'tmax', 'dmax', 'tmin', 'dmin',
#       'hmdy', 'hmax', 'hmin', 'wdsp', 'wdct', 'gust']
bools_only = weather_data.select_dtypes(include=['bool']) # no booleans then

#histograms:
plt.hist(numbers_only['wsid']) # majority between 300 and 423
max(numbers_only['wsid'])

plt.hist(numbers_only['elvt']) # skewed left
max(numbers_only['elvt'])

plt.hist(numbers_only['lat']) # skewed left
max(numbers_only['lat'])

plt.hist(numbers_only['lon'])
max(numbers_only['lon'])

plt.hist(numbers_only['yr']) #2002 to 2016
max(numbers_only['yr'])

plt.hist(numbers_only['mo'])#peaks at begin and end of year
max(numbers_only['mo'])

plt.hist(numbers_only['da'])
max(numbers_only['da'])

plt.hist(numbers_only['hr']) #peaks every 8 hours?
max(numbers_only['hr'])

plt.hist(numbers_only['prcp'])
max(numbers_only['prcp'])

plt.hist(numbers_only['stp'])
max(numbers_only['stp'])

plt.hist(numbers_only['smax'])
max(numbers_only['smax'])

plt.hist(numbers_only['smin'])#right
max(numbers_only['smin'])

plt.hist(numbers_only['gbrd'])#left
max(numbers_only['gbrd'])

plt.hist(numbers_only['temp'])#normal
max(numbers_only['temp'])

plt.hist(numbers_only['dewp'])#normal
max(numbers_only['dewp'])

plt.hist(numbers_only['tmax'])#normal
max(numbers_only['tmax'])

plt.hist(numbers_only['dmax'])#normal
max(numbers_only['dmax'])

plt.hist(numbers_only['tmin'])#normal
max(numbers_only['tmin'])

plt.hist(numbers_only['dmin'])#normal
max(numbers_only['dmin'])

plt.hist(numbers_only['hmdy'])#right skew
max(numbers_only['hmdy'])

plt.hist(numbers_only['hmax'])#right skew
max(numbers_only['hmax'])

plt.hist(numbers_only['hmin'])#right skew
max(numbers_only['hmin'])

plt.hist(numbers_only['wdsp'])# left skew
max(numbers_only['wdsp'])

plt.hist(numbers_only['wdct'])# left skew
max(numbers_only['wdct'])

plt.hist(numbers_only['gust']) # left skew
max(numbers_only['gust'])


strings_only = weather_data.select_dtypes(include=['object'])
print(strings_only.columns)
#'wsnm', 'inme', 'city', 'prov', 'mdct', 'date']

plt.hist(strings_only['wsnm']) #good spread?
max(strings_only['wsnm'])

plt.hist(strings_only['inme']) #good spread


plt.hist(strings_only['city']) #good spread
max(strings_only['city'])

plt.hist(strings_only['prov'])  #four main categories


plt.hist(strings_only['mdct']) #Good spread


plt.hist(strings_only['date']) #skewed, many

###########################################
# NUMBER ONE: CHECK FOR NULLS
###########################################
numbers_only.isna().sum()
#wsid          0
# elvt          0
# lat           0
# lon           0
# yr            0
# mo            0
# da            0
# hr            0
# prcp    8371184
# stp           0
# smax          0
# smin          0
# gbrd    4108820
# temp         31
# dewp        475
# tmax         26
# dmax        310
# tmin         34
# dmin        807
# hmdy          0
# hmax         12
# hmin         44
# wdsp     925561
# wdct          0
# gust     316474

df_1 = weather_data
df_2 = df_1.dropna(subset=['temp', 'dewp', 'tmax', 'dmax', 'tmin', 'dmin', 'hmax', 'hmin'])
df_2.isna().sum()
#prcp    8369669
#gbrd    4108378
#wdsp     925418
#gust     316427
plt.hist(df_2['prcp'])
plt.hist(df_2['gbrd'])
plt.hist(df_2['wdsp'])
plt.hist(df_2['gust'])
#quite a bit of skew, going to do Median imputation
df_3 = df_2
pd.options.mode.chained_assignment = None

df_4 = df_3.fillna(df_3.median())
# df_3.loc['prcp'] = df_2['prcp'].fillna(df_2['prcp'].median())
# df_3.loc['gbrd'] = df_2['gbrd'].fillna(df_2['gbrd'].median())
# df_3.loc['wdsp'] = df_2['wdsp'].fillna(df_2['wdsp'].median())
# df_3.loc['gust'] = df_2['gust'].fillna(df_2['gust'].median())
df_4.isna().sum()
df_4=df_4.dropna()
#perfect

###########################################
# NUMBER TWO: CHECK FOR OUTLIERS
###########################################
num_cols1 = ['wsid', 'elvt', 'da', 'hr', 'prcp', 'stp',
       'smax', 'smin', 'gbrd', 'temp', 'dewp', 'tmax', 'dmax', 'tmin', 'dmin',
       'hmdy', 'hmax', 'hmin', 'wdsp', 'wdct', 'gust']

#'lat', 'lon', 'yr', 'mo' dont remove these though!

df_outliers = df_4
df_outliers.isna().sum()

for i in num_cols1:
       q1 = np.percentile(df_outliers[i],25,interpolation='midpoint')
       q3 = np.percentile(df_outliers[i],75,interpolation='midpoint')
       iqr = q3 - q1

       upper = q3 + 1.5 * iqr
       lower = q1 - 1.5 * iqr
       print('I is', i, upper, lower)
       print('first', df_outliers.shape)


       df_outliers_new = df_outliers[(df_outliers[i]>= lower) & (df_outliers[i]<= upper)]
       print('last', df_outliers_new.shape)
       df_outliers = df_outliers_new.copy()
       #this operation should allow for the outliers to be taken and removed from each column of
       # the data frame, and then the data frame to update as a whole throughout

df_outliers_new.isna().sum()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

num_cols = ['wsid', 'elvt', 'prcp', 'stp',
       'smax', 'smin', 'gbrd', 'temp', 'dewp', 'tmax', 'dmax', 'tmin', 'dmin',
       'hmdy', 'hmax', 'hmin', 'wdsp', 'wdct', 'gust']

df_5=df_outliers
df_5.isna().sum()

df_5[['wsid', 'elvt', 'prcp', 'stp',
       'smax', 'smin', 'gbrd', 'temp', 'dewp', 'tmax', 'dmax', 'tmin', 'dmin',
       'hmdy', 'hmax', 'hmin', 'wdsp', 'wdct', 'gust']] = scaler.fit_transform(df_5[['wsid', 'elvt', 'prcp', 'stp',
       'smax', 'smin', 'gbrd', 'temp', 'dewp', 'tmax', 'dmax', 'tmin', 'dmin',
       'hmdy', 'hmax', 'hmin', 'wdsp', 'wdct', 'gust']])
#review this to see if it normalized as anticipated
#THEN Reconnect to non numerics, all done!

print('scaled')


##########################################
# NUMBER THREE: CHECK FOR DUPLICATES
##########################################
dups = weather_data[weather_data.duplicated()]
print(dups) # no duplicates

df_dup = df_5[df_5.duplicated()]
print(df_dup) # no duplicates


##########################################
# NUMBER FOUR: ERRONEOUS DATA (Doesn't seem to make sense with the set
##########################################

#do the mins and maxes make sense?
desc_df = pd.DataFrame(weather_data.describe())
# The maximums and minimums make sense for the data set, another check would be to see if the
# text strings make sense, or if there is a "business reason" that any of this is odd


#######LABEL ENCODING############
# Drop date related, not needed any more: 'mdct', 'date'
df_6 = df_5.drop('date', axis=1)
df_6 = df_6.drop('mdct', axis=1)

#'wsnm', 'inme', 'city', 'prov']
final_df = df_6
le = LabelEncoder()
final_df['wsnm'] = le.fit_transform(final_df['wsnm'])
final_df['inme'] = le.fit_transform(final_df['inme'])
final_df['city'] = le.fit_transform(final_df['city'])
final_df['prov'] = le.fit_transform(final_df['prov'])

y = final_df.loc[:, 'temp']  # could be gust too, which would be interesting
X = final_df.loc[:, final_df.columns!= 'temp']

X.isna().sum()
######### DIMensionality reduction for fun############
from sklearn.decomposition import PCA

pca = PCA()
pca.fit_transform(X)
pca_variance = pca.explained_variance_

plt.figure(figsize=(8, 6))
plt.bar(range(22), pca_variance, alpha=0.5, align='center', label='individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal components')
plt.show()
# 16 looks really good!


pca2 = PCA(n_components=16)
pca2.fit(X)
X_mod = pca2.transform(X)


###############Modeling###############
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

######### Decision Tree First:

dtr = DecisionTreeRegressor()

dtr = dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)

# METRICS:
from sklearn.metrics import r2_score, mean_squared_error
print('r2', r2_score(y_test, y_pred))
print('MSE', mean_squared_error(y_test, y_pred))
#r2 0.9955072375694298
#MSE 0.00014458571720858828


######### Random Forest Reg:
rfr = DecisionTreeRegressor()
rfr = rfr.fit(X_train, y_train)
y_pred2 = rfr.predict(X_test)

# METRICS
print('r2', r2_score(y_test, y_pred2))
print('MSE', mean_squared_error(y_test, y_pred2))
#r2 0.9955132412522315
#MSE 0.00014439250717418535



##########
##NOW WITH DIMENSION REDUCTION
#########

X_train, X_test, y_train, y_test = train_test_split(X_mod, y, test_size=0.2, random_state=42)

######### Decision Tree First:

dtr = DecisionTreeRegressor()

dtr = dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)

# METRICS:
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, f1_score
print('r2', r2_score(y_test, y_pred))
print('Accuracy', accuracy_score(y_test, y_pred))
print('Precision', precision_score(y_test, y_pred))
print('F1 Score', f1_score(y_test, y_pred))



######### Random Forest Reg:
rfr = DecisionTreeRegressor()
rfr = rfr.fit(X_train, y_train)
y_pred2 = rfr.predict(X_test)

# METRICS
print('r2', r2_score(y_test, y_pred2))
print('Accuracy', accuracy_score(y_test, y_pred2))
print('Precision', precision_score(y_test, y_pred2))
print('F1 Score', f1_score(y_test, y_pred2))