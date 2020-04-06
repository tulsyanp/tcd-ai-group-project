from time import time
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def fetch_dataset():
    dataset = fetch_lfw_people(min_faces_per_person=100)   # labelled faces in the wild data with users more than 100 faces

    return dataset


def fetch_data_details(dataset):
    n_samples, height, width = dataset.images.shape

    X = dataset.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = dataset.target
    target_names = dataset.target_names
    n_classes = target_names.shape[0]

    print("Total DATASET size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    return n_samples, height, width, X, n_features, y, target_names, n_classes


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test


def dimensionality_reduction_PCA(n_components, X_train, height, width):
    print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, height, width))

    return pca, eigenfaces


def dimensionality_reduction_ICA(n_components, X_train, height, width):
    print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
    t0 = time()
    ica = FastICA(n_components=n_components, whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    eigenfaces = ica.components_.reshape((n_components, height, width))

    return ica, eigenfaces


def train_text_transform_Model(model, X_train, X_test):
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_model = model.transform(X_train)
    X_test_model = model.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    return X_train_model, X_test_model


def classification_svc(X_train_model, y_train):
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_model, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    return clf


def prediction(model, data):
    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = model.predict(data)
    print("done in %0.3fs" % (time() - t0))

    return y_pred


def print_report(y_test, y_pred, target_names, n_classes):
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


def plot_images(images, titles, height, width, n_row=1, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((height, width)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

    plt.show()


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


# def plot_confusion_matrix(y_true, y_pred, matrix_title):
#     """confusion matrix computation and display"""
#     plt.figure(figsize=(9, 9), dpi=100)
#
#     # use sklearn confusion matrix
#     cm_array = confusion_matrix(y_true, y_pred)
#     plt.imshow(cm_array[:-1, :-1], interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title(matrix_title, fontsize=16)
#
#     cbar = plt.colorbar(fraction=0.046, pad=0.04)
#     cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
#
#     true_labels = np.unique(y_true)
#     pred_labels = np.unique(y_pred)
#     xtick_marks = np.arange(len(true_labels))
#     ytick_marks = np.arange(len(pred_labels))
#
#     plt.xticks(xtick_marks, true_labels, rotation=90)
#     plt.yticks(ytick_marks, pred_labels)
#     plt.tight_layout()
#     plt.ylabel('True label', fontsize=14)
#     plt.xlabel('Predicted label', fontsize=14)
#     plt.tight_layout()
#
#     plt.show()
#
#
#
# plot_confusion_matrix(y_test, y_pred, "matriz")