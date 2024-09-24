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

dataset = read_csv(filename, names=columns, sep=',')

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


