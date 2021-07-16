import numpy as np
from sklearn import svm

from process_email import process_email
from email_features import email_features
from get_vocabulary_dict import get_vocabulary_dict


def read_file(file_path: str) -> str:
    """Return the content of the text file under the given path.

    :param file_path: path to the file
    :return: file content
    """

    with open(file_path, 'r') as f:
        file_content = f.read()
        return file_content


print('\nPreprocessing sample email (emailSample1.txt)\n')

file_contents = read_file('data/emailSample1.txt')
word_indices = process_email(file_contents)

# Print Stats
print('Word Indices: \n')
print(word_indices)
print('\n\n')

# input('Program paused. Press enter to continue.\n')

print('\nExtracting features from sample email (emailSample1.txt)\n')

# Extract Features
file_contents = read_file('data/emailSample1.txt')
word_indices = process_email(file_contents)
features = email_features(word_indices)

# Print Stats
print('Length of feature vector: {}\n'.format(len(features[0])))
print('Number of non-zero entries: {}\n'.format(sum(sum(features > 0))))

# input('Program paused. Press enter to continue.\n')

# Load the Spam Email dataset

print('\nLoading the training dataset...')
X_train = np.genfromtxt('data/spamTrain_X.csv', delimiter=',')
y_train = np.genfromtxt('data/spamTrain_y.csv', delimiter=',')
print('The training dataset was loaded.')

print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')

# Create a linear SVC classifier (with C = 0.1).
clf = svm.SVC(kernel='linear', C=0.1, random_state=1, probability=True)

# Fit the SVC model using the training data.
clf.fit(X_train, y_train)

# Predict the labelling.
y_pred = clf.predict(X_train)

# Compute the training accuracy.
acc_train = np.mean(y_pred == y_train)
print('Training Accuracy: {:.2f}%\n'.format(acc_train * 100))

# Load the test dataset ('data/spamTest_X.csv', 'data/spamTest_y.csv').
X_test = np.genfromtxt('data/spamTest_X.csv', delimiter=',')
y_test = np.genfromtxt('data/spamTest_y.csv', delimiter=',')

print('\nEvaluating the trained Linear SVM on a test set ...\n')

# Predict the labelling.
y_pred = clf.predict(X_test)

# Compute the training accuracy.
acc_test = np.mean(y_pred == y_test)
print('Test Accuracy: {:.2f}%\n'.format(acc_test * 100))

# input('Program paused. Press enter to continue.\n')

weights = clf.coef_.reshape(-1)
idx = np.argsort(-weights)

vocabulary_dict = get_vocabulary_dict()

print('\nTop predictors of spam: \n')
for i in range(15):
    print(' {word:<20}: {weight:10.6f}'.format(word=vocabulary_dict[idx[i]], weight=weights[idx[i]]))

print('\n\n')
# input('\nProgram paused. Press enter to continue.\n')

filename = 'data/spamSample1.txt'

# Read and predict
file_contents = read_file(filename)
word_indices = process_email(file_contents)
x = email_features(word_indices)

# Predict the labelling.
y_pred = clf.predict(X_test)

print('\nProcessed {}\n\nSpam Classification: {}\n'.format(filename, y_pred[0] > 0))
print('(1 indicates spam, 0 indicates not spam)\n\n')
