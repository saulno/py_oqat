import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from config_algorithms import ACOConfig
from oqat import OQATClassifier, OQATModel
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix

def pretty_print_confusion_matrix(confusion_matrix):
    for row in confusion_matrix:
        print(row)

# Read csv file and load it into a numpy array
# df = pd.read_csv('datasets/test2.csv')
df = pd.read_csv('datasets/hayes_roth.csv')

# Separate features and labels into two dataframes
X = df.drop('class', axis=1)
y = df['class']

# Define the feature type for each column (discrete or continuous)
column_names = X.columns.to_list()
column_types = ['cat', 'cat', 'cat']

# transform the dataframes into a numpy array
X = X.values
y = y.values

# Transform the string labels into integers
X = preprocessing.OrdinalEncoder().fit_transform(X)
y = preprocessing.LabelEncoder().fit_transform(y)

# Make cross validation experiments to compare different algorithms
# number_of_experiments = 10
# for i in range(number_of_experiments):



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Selelct the entries that belong to the class 2
x_2 = X_test[y_test == 2]
print(x_2)

# Balance the training set
# print("Before SMOTE")
# print(pd.Series(y_train).value_counts())
# smote = SMOTE()
# X_train, y_train = smote.fit_resample(X_train, y_train)
# print("After SMOTE")
# print(pd.Series(y_train).value_counts())

# Run a classification tree algorithm from sklearn
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
cf = confusion_matrix(y_test, y_pred)
print("Decision Tree")
pretty_print_confusion_matrix(cf)
print("Score: ", tree_model.score(X_test, y_test))
# text_repr = tree.export_text(tree_model)
# print(text_repr)

# # Run the OQAT algorithm
aco_config = ACOConfig(algorithm="vertex-ac", cycles=20, ants=10, alpha=1, rho=0.99, tau_max=6., tau_min=0.01)
classifier = OQATClassifier(collision_strategy="random", heuristic="aco", heuristic_config=aco_config)
classifier.fit(X_train, y_train, column_names, column_types)
print(classifier.model)
y_pred = classifier.predict(X_test, column_names)
print(y_pred)
cf = classifier.confusion_matrix(y_pred, y_test)
print("OQAT")
pretty_print_confusion_matrix(cf)
print("Score: ", classifier.score(y_pred, y_test))