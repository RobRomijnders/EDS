import numpy as np
from sklearn.linear_model import LogisticRegression

train_data = np.genfromtxt('incomplete_train_data.csv', delimiter=',', skip_header=1, usecols=(1,2,3,4,5,6,7,8,9))
test_data = np.genfromtxt('incomplete_test_data.csv', delimiter=',', skip_header=1, usecols=(1,2,3,4,5,6,7,8,9))

X_train, y_train = train_data[:, 1:], train_data[:, 0]
X_test, y_test = test_data[:, 1:], test_data[:, 0]

print('training samples: %3i  testing samples %3i' % (len(y_train), len(y_test)))
print('X_train has %3i missing values' % np.isnan(X_train).sum())
print('X_test has %3i missing values' % np.isnan(X_test).sum())

# make your own imputation method
# example: random imputation from observed values
def fit_imputation(incomplete_data):

    possible_imputations = []
    for i in range(incomplete_data.shape[1]):
        possible_imputations.append(incomplete_data[~np.isnan(incomplete_data[:, i]),i])

    return possible_imputations

def apply_imputation(incomplete_data, possible_imputations):

    complete_data = incomplete_data
    for i in range(len(possible_imputations)):
        n_missing = np.isnan(incomplete_data[:, i]).sum()
        imputations = np.random.choice(possible_imputations[i], size=n_missing, replace=True)
        complete_data[np.isnan(incomplete_data[:, i]),i] = imputations

    return complete_data

# make sure you fit on your train set
possible_imputations = fit_imputation(X_train)
# and apply first on train set, then on test set
X_train = apply_imputation(X_train, possible_imputations)
X_test = apply_imputation(X_test, possible_imputations)

print('X_train has %3i missing values' % np.isnan(X_train).sum())
print('X_test has %3i missing values' % np.isnan(X_test).sum())

model = LogisticRegression().fit(X_train, y_train)
print('\nSimple logistic regression with random imputation gives score:')
print(model.score(X_test, y_test))