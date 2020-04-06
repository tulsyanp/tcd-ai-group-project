from main import fetch_dataset, fetch_data_details, split_data, dimensionality_reduction_LDA, train_text_transform_LDA, classification_svc, prediction, print_report, plot_images, title, plot_images_lda

# Load data
dataset = fetch_dataset()

# get dataset details and target names
n_samples, height, width, X, n_features, y, target_names, n_classes = fetch_data_details(dataset)

# split into a training and testing set
X_train, X_test, y_train, y_test = split_data(X, y)

# compute LDA
n_components = 150

lda, pca = dimensionality_reduction_LDA(n_components, X_train, y_train)

X_train_lda, X_test_lda = train_text_transform_LDA(lda, pca, X_train, X_test)

# Training a SVM classification model
clf = classification_svc(X_train_lda, y_train)

# Quantitative evaluation of the model quality on the test set
y_pred = prediction(clf, X_test_lda)

# printing classification report
print_report(y_test, y_pred, target_names, n_classes)

# printing images
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_images(X_test, prediction_titles, height, width)

# plot fisherfaces
fisherface_titles = ["fisherface %d" % i for i in range(4)]
plot_images_lda(pca, lda, fisherface_titles, height, width)