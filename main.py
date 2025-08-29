from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = datasets.load_wine()
X = iris.data
y = iris.target

# Scale data
X = StandardScaler().fit(X).transform(X)
