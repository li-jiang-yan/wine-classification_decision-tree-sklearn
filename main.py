from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Scale data
X = StandardScaler().fit(X).transform(X)

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Train model
clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_train, y_train)

# Classify test data using model
y_pred = clf.predict(X_test)

# Plot the decision tree model trained
plot_tree(clf, feature_names=wine.feature_names)
plt.show()
