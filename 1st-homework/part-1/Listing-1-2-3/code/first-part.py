# Load libraries

from pandas import read_csv

from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

# Load dataset
filename = 'glass/glass.data'
names = ['Id', 'Refractive Index(RI)', 
         'Sodium(Na)', 'Magnesium(Mg)', 
         'Aluminum(Al)', 'Silicon(Si)', 
         'Potassium(K)', 'Calcium(Ca)', 
         'Barium(Ba)', 'Iron(Fe)', 'glassType']

dataset = read_csv(filename, names=names)

dataset = dataset.drop(['Id'], axis =1)


# Summarize Data
# shape
print(dataset.shape)

# print the first 10 rows of the data
print(dataset.head(10))

# Descriptive statistics
print(dataset.describe())

# class distribution by glass ttype
print (dataset.groupby('glassType').size())


dataset = dataset.drop(['glassType'], axis =1)

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(3,3), figsize=(9, 9), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist(figsize=(9, 9))
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset, figsize =(18,18))
pyplot.show()
