# loading all the needed libraries
from matplotlib import pyplot
import numpy
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from numpy import set_printoptions 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

# loading the dataset
filename = 'C:\\Users\\fabio\\Desktop\\glass+identification\\glass.data'
columns = ['Id', 'Ri', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
columns_after_drop = ['Ri', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']


"""

For a better understanding of the abbreviations:
1. Id number: 1 to 214
   2. RI: refractive index
   3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as 
                  are attributes 4-10)
   4. Mg: Magnesium
   5. Al: Aluminum
   6. Si: Silicon
   7. K: Potassium
   8. Ca: Calcium
   9. Ba: Barium
  10. Fe: Iron
  11. Type of glass: (class attribute)
      -- 1 building_windows_float_processed
      -- 2 building_windows_non_float_processed
      -- 3 vehicle_windows_float_processed
      -- 4 vehicle_windows_non_float_processed (none in this database)
      -- 5 containers
      -- 6 tableware
      -- 7 headlamps
"""

dataset_before_drop = read_csv(filename, names=columns, sep=',')
dataset = dataset_before_drop.drop(['Id', 'Type'], axis=1)

print("haad:" , dataset.head())

print("Shape of dataset:", dataset.shape)

# Set pandas display options for better output visibility
set_option('display.max_rows', 500)
set_option('display.width', 100)
set_option('display.precision', 3)  # Corrected option

# Print dataset types to check data types of each column
print("Data types of each column:\n", dataset.dtypes)

# Show the first 20 rows of the dataset
print("First 20 rows of the dataset:\n", dataset.head(20))

print(dataset.describe())

print(dataset.groupby('Ri').size())

print(dataset.corr(method = 'pearson'))

print(dataset.skew())

print(dataset.hist())
pyplot.figsize = (8,8)
pyplot.savefig('histograms.png', dpi=300)
pyplot.show()

dataset.plot(kind='density' , subplots=True, layout=(4,4), sharex=False, figsize=(14,14))
pyplot.show()

dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, figsize=(14,14))
pyplot.show()

fig = pyplot.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(columns_after_drop)
ax.set_yticklabels(columns_after_drop)
pyplot.show()

fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()

scatter_matrix(dataset)
pyplot.figure(figsize=(20,18))
pyplot.show()

array = dataset.values
X = array[:, 0:9]  # Select all 9 columns for X (all features)
scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize the scaler
rescaledX = scaler.fit_transform(X)  # Scale the feature data
set_printoptions(precision=3)  # Set print precision
print(rescaledX[0:5, :])  # Print the first 5 rows of scaled data






print(dataset[['Ba', 'Fe']].describe())

"""
Zeros in the Output: The presence of many zeros in the original data for both Ba and Fe leads to many zeros in the scaled output rescaledX because the scaling does not change values that are already zero.

Interpretation of Zeros: This is expected behavior and indicates that a significant number of samples for both features have a value of zero. This may suggest that these features are not contributing significantly to variance in your datasetZeros in the Output: The presence of many zeros in the original data for both Ba and Fe leads to many zeros in the scaled output rescaledX because the scaling does not change values that are already zero.

Interpretation of Zeros: This is expected behavior and indicates that a significant number of samples for both features have a value of zero. This may suggest that these features are not contributing significantly to variance in your dataset

"""


X = array[:, 0:9]  # Select all 9 columns for X (all features)
scaler_standard = StandardScaler().fit(X)
rescaled_standardX = scaler_standard.transform(X)  # Scale the feature data
set_printoptions(precision=3)  # Set print precision
print(rescaled_standardX[0:5, :])  # Print the first 5 rows of scaled data

X = array[:, 0:9]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
set_printoptions(precision=3)
print(normalizedX[0:5, :])


binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
set_printoptions(precision=3)
print(binaryX[0:5, :])
