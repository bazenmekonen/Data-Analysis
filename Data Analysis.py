import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Data Pre-processing
def load_and_preprocess_data(ip2ttt_train):
    data = pd.read_csv("C:\\Users\\Bazen Mekonen\\Downloads\\ip2ttt_train.data", header=None)

    # One-hot encode categorical features
    one_hot_encoder = OneHotEncoder()
    X = one_hot_encoder.fit_transform(data.iloc[:, :-1]).toarray()

    # Convert labels to integers
    y = data.iloc[:, -1].astype('category').cat.codes

    return X, y


train_X, train_y = load_and_preprocess_data('ip2ttt_train.data')
valid_X, valid_y = load_and_preprocess_data('ip2ttt_valid.data')
test_X, test_y = load_and_preprocess_data('ip2ttt_test.data')

# K-NN
print("K-NN results:")
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_X, train_y)
    print(f"K = {k}:")
    print("Training accuracy:", accuracy_score(train_y, knn.predict(train_X)))
    print("Validation accuracy:", accuracy_score(valid_y, knn.predict(valid_X)))
    print("Test accuracy:", accuracy_score(test_y, knn.predict(test_X)))

# Logistic Regression
print("Logistic Regression results:")
for penalty in ['l1', 'l2']:
    for solver in ['newton-cg', 'lbfgs']:
        # Skip invalid combinations
        if penalty == 'l1' and solver in ['newton-cg', 'lbfgs']:
            continue
        logreg = LogisticRegression(penalty=penalty, solver=solver, max_iter=1000)
        logreg.fit(train_X, train_y)
        print(f"Penalty = {penalty}, Solver = {solver}:")
        print("Training accuracy:", accuracy_score(train_y, logreg.predict(train_X)))
        print("Validation accuracy:", accuracy_score(valid_y, logreg.predict(valid_X)))
        print("Test accuracy:", accuracy_score(test_y, logreg.predict(test_X)))

# Decision Trees
print("Decision Trees results:")
for min_samples_leaf in range(1, 11):
    dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=min_samples_leaf)
    dt.fit(train_X, train_y)
    print(f"Min samples leaf = {min_samples_leaf}:")
    print("Training accuracy:", accuracy_score(train_y, dt.predict(train_X)))
    print("Validation accuracy:", accuracy_score(valid_y, dt.predict(valid_X)))
    print("Test accuracy:", accuracy_score(test_y, dt.predict(test_X)))

# Random Forests
print("Random Forests results:")
for max_depth in [3, 5]:
    for min_samples_leaf in [5, 10]:
        rf = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        rf.fit(train_X, train_y)
        print(f"Max depth = {max_depth}, Min samples leaf = {min_samples_leaf}:")
print("Training accuracy:", accuracy_score(train_y, rf.predict(train_X)))
print("Validation accuracy:", accuracy_score(valid_y, rf.predict(valid_X)))
print("Test accuracy:", accuracy_score(test_y, rf.predict(test_X)))