# --- Import libraries ---
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from keras.layers import Dense, GRU, LSTM, SimpleRNN
from keras.models import load_model, Sequential
from my_class import DataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

sns.set_style("whitegrid")


# --- Functions ---
def get_train_test(df):
    """
    Split data frame into training and test sets

    Arguments
    ---------
    :param df:  phrase data frame

    Return
    ------
    :return:    training and test sets
    """

    # clean the data frame
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    # train test split 0 - negative phrases
    neg = df[df.Sentiment == 0]
    neg.reset_index(inplace=True, drop=True)
    neg_train, neg_test = train_test_split(neg, test_size=0.2, random_state=3452)
    neg_train.reset_index(inplace=True, drop=True)
    neg_test.reset_index(inplace=True, drop=True)

    # train test split 1 - somewhat negative phrases
    some_neg = df[df.Sentiment == 1]
    some_neg.reset_index(inplace=True, drop=True)
    some_neg_train, some_neg_test = train_test_split(some_neg, test_size=0.2, random_state=5642)
    some_neg_train.reset_index(inplace=True, drop=True)
    some_neg_test.reset_index(inplace=True, drop=True)

    # train test split 2 - neutral phrases
    neu = df[df.Sentiment == 2]
    neu.reset_index(inplace=True, drop=True)
    neu_train, neu_test = train_test_split(neu, test_size=0.2, random_state=4562)
    neu_train.reset_index(inplace=True, drop=True)
    neu_test.reset_index(inplace=True, drop=True)

    # train test split 3 - somewhat positive phrases
    some_pos = df[df.Sentiment == 3]
    some_pos.reset_index(inplace=True, drop=True)
    some_pos_train, some_pos_test = train_test_split(some_pos, test_size=0.2, random_state=2134)
    some_pos_train.reset_index(inplace=True, drop=True)
    some_pos_test.reset_index(inplace=True, drop=True)

    # train test split 4 - positive phrases
    pos = df[df.Sentiment == 4]
    pos.reset_index(inplace=True, drop=True)
    pos_train, pos_test = train_test_split(pos, test_size=0.2, random_state=8456)
    pos_train.reset_index(inplace=True, drop=True)
    pos_test.reset_index(inplace=True, drop=True)

    # create training set
    training = pd.concat([neg_train, some_neg_train, neu_train, some_pos_train, pos_train], ignore_index=True)
    training = shuffle(training, random_state=3642)
    training.reset_index(inplace=True, drop=True)

    # create test set
    test = pd.concat([neg_test, some_neg_test, neu_test, some_pos_test, pos_test], ignore_index=True)
    test = shuffle(test, random_state=2374)
    test.reset_index(inplace=True, drop=True)

    # return
    return training, test


def vectorize_phrases(phrases, phrase_type, is_dl=False):
    """
    Vectorize phrases

    Arguments
    ---------
    :param phrases:         phrases to vectorize

    :param phrase_type:     name of processed phrases used
                            can choose ["norm", "stemmed", "lemmed"]

    Optional Arguments
    ------------------
    :param is_dl:           is using a deep learning model (ann, rnn, lstm, gru)?

    Return
    ------
    :return:                vectorized phrases
    """

    # create and fit vectorizer
    if os.path.exists(
            "C:\\Users\\15713\\Desktop\\DS Projects\\Sentiment Analysis on Movie "
            "Reviews\\sentiment-analysis-on-movie-reviews\\models\\" + phrase_type + "_vectorizer.pickle"):
        # load vectorizer
        save_vectorizer = open(
            "C:\\Users\\15713\\Desktop\\DS Projects\\Sentiment Analysis on Movie "
            "Reviews\\sentiment-analysis-on-movie-reviews\\models\\" + phrase_type + "_vectorizer.pickle",
            "rb")
        vectorizer = pickle.load(save_vectorizer)
        save_vectorizer.close()
    else:
        # fit vectorizer
        vectorizer = TfidfVectorizer()
        vectorizer.fit(phrases)
        # save vectorizer
        save_vectorizer = open(
            "C:\\Users\\15713\\Desktop\\DS Projects\\Sentiment Analysis on Movie "
            "Reviews\\sentiment-analysis-on-movie-reviews\\models\\" + phrase_type + "_vectorizer.pickle",
            "wb")
        pickle.dump(vectorizer, save_vectorizer)
        save_vectorizer.close()

    # vectorize phrases
    vectorized_phrases = vectorizer.transform(phrases)

    # save vectorized phrases (training data only)
    # to be used with my_class.py for DL models
    # not needed if using batch_generator in, perhaps, future projects
    if is_dl:
        if len(os.listdir("C:\\Users\\15713\\Desktop\\DS Projects\\Sentiment Analysis on Movie "
                          "Reviews\\sentiment-analysis-on-movie-reviews\\data\\" + phrase_type)) == 0:
            vectorized_array = vectorized_phrases.toarray()
            for i in range(vectorized_array.shape[0]):
                np.save("C:\\Users\\15713\\Desktop\\DS Projects\\Sentiment Analysis on Movie "
                        "Reviews\\sentiment-analysis-on-movie-reviews\\data\\" + phrase_type + "\\id-" + str(i + 1) +
                        ".npy", vectorized_array[i])

    # return
    return vectorized_phrases


def build_ML_model(X_train, y_train, phrase_type, model_type, k_fold=10):
    """
    Build the best ML model (depending on the chosen arguments) to classify phrases - only use One Vs. Rest Classifiers
    (multi-class: negative, somewhat negative, neutral, somewhat positive, positive)

    Arguments
    ---------
    :param X_train:             phrases to train on (X in training data)

    :param y_train:             sentiments to train on (y in training data)

    :param phrase_type:         name of processed phrases used
                                can choose ["norm", "stemmed", "lemmed"]

    :param model_type:          which ML model to build
                                can choose ["logistic" (Logistic Regression),
                                            "decision_tree" (Decision Tree),
                                            "random_forest" (Random Forest)]

    Optional Arguments
    ------------------
    :param k_fold:              number of folds (k-fold cross-validation)
                                default to 10

    Return
    ------
    :return:                    the best ML model depending on the chosen arguments
    """

    # binarize y_train (needed for One Vs. Rest classifiers)
    y_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4])

    # get model
    models = {"logistic": LogisticRegression(random_state=2374, n_jobs=-1),
              "decision_tree": DecisionTreeClassifier(random_state=8743, min_samples_leaf=5),
              "random_forest": RandomForestClassifier(random_state=2947, min_samples_leaf=5,
                                                      max_features="sqrt", n_jobs=-1)}
    model = models[model_type]

    # vectorize phrases
    vectorized_X_train = vectorize_phrases(phrases=X_train, phrase_type=phrase_type)

    # set parameters for grid search
    param_grid = {}
    if model_type == "logistic":
        param_grid["estimator__C"] = [0.01, 0.1, 1.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 100.0]
        param_grid["estimator__solver"] = ["saga", "sag", "lbfgs"]
        param_grid["estimator__warm_start"] = [True, False]
    elif model_type == "decision_tree":
        param_grid["estimator__criterion"] = ["gini", "entropy"]
        param_grid["estimator__min_samples_split"] = [10, 25]
        param_grid["estimator__max_features"] = [None, "sqrt"]
        param_grid["estimator__class_weight"] = ["balanced", None]
    else:
        param_grid["estimator__criterion"] = ["gini", "entropy"]
        param_grid["estimator__min_samples_split"] = [10, 25]
        param_grid["estimator__class_weight"] = ["balanced", None]

    # create and fit grid search
    kfold = KFold(n_splits=k_fold, random_state=5765)
    clf = GridSearchCV(OneVsRestClassifier(model, n_jobs=-1), param_grid, cv=kfold, scoring="roc_auc", n_jobs=-1)
    clf.fit(vectorized_X_train, y_train)

    # evaluate grid search
    print("Best estimator:")
    print(clf.best_estimator_)
    print()

    print("Best parameters:")
    print(clf.best_params_)
    print()

    print("Best score:")
    print(clf.best_score_)
    print()

    print("Grid scores on training set:")
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    params_sets = clf.cv_results_["params"]
    for mean, std, params in zip(means, stds, params_sets):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    # save classifier
    save_classifier = open(
        "C:\\Users\\15713\\Desktop\\DS Projects\\Sentiment Analysis on Movie "
        "Reviews\\sentiment-analysis-on-movie-reviews\\models\\" + phrase_type + "_" + model_type + "_model.pickle",
        "wb")
    pickle.dump(clf, save_classifier)
    save_classifier.close()

    # return
    return clf


# not used in this project but a good resource
# source: https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-9-neural-networks-with-tfidf-vectors-using-d0b4af6be6d7
# def batch_generator(X_data, y_data, batch_size):
#     samples_per_epoch = X_data.shape[0]
#     number_of_batches = samples_per_epoch / batch_size
#     counter = 0
#     index = np.arange(np.shape(y_data)[0])
#     while 1:
#         index_batch = index[batch_size * counter:batch_size * (counter + 1)]
#         X_batch = X_data[index_batch, :].toarray()
#         y_batch = y_data[index_batch, :]
#         counter += 1
#         yield X_batch, y_batch
#         if counter > number_of_batches:
#             counter = 0


def change_key(key):
    """
    Change key format for label dictionary (from "0" to "id-1," from "1" to "id-2," and so on)

    Arguments
    ---------
    :param key:     original key

    Return
    ------
    :return:        new key
    """

    # return
    return "id-" + str(key + 1)


def get_partition_and_labels(y_train):
    """
    Create partition and label dictionaries

    Arguments
    ---------
    :param y_train:     sentiments to train on (y in training data)

    Return
    ------
    :return:            partition and label dictionaries
    """

    # create partition dictionary with keys train and validation, each contains a list of ids
    partition = {"train": ["id-" + str(i + 1) for i in range(99875)],
                 "validation": ["id-" + str(i + 1 + 99875) for i in range(24969)]}

    # create label dictionary
    labels = y_train.to_dict()
    labels = {change_key(k): v for k, v in labels.items()}

    # return
    return partition, labels


def build_DL_model(X_train, y_train, phrase_type, model_type):
    """
    Build a simple DL model (depending on the chosen arguments) to classify phrases

    Arguments
    ---------
    :param X_train:         phrases to train on (X in training data)

    :param y_train:         sentiments to train on (y in training data)

    :param phrase_type:     name of processed phrases used
                            can choose ["norm", "stemmed", "lemmed"]

    :param model_type:      which DL model to build
                            can choose ["ann" (Artificial Neural Network),
                                        "rnn" (Recurrent Neural Network),
                                        "lstm" (Long Short-Term Memory),
                                        "gru" (Gated Recurrent Unit)]

    Return
    ------
    :return:                a simple DL model depending on the chosen arguments
    """

    # vectorize phrases
    vectorized_X_train = vectorize_phrases(phrases=X_train, phrase_type=phrase_type, is_dl=True)

    # set up data and labels
    partition, labels = get_partition_and_labels(y_train)

    # set up generators
    per_epoch = 512
    if model_type == "ann" and phrase_type == "norm":
        recurrent = False
        batch_size = 512
    elif model_type == "ann" and phrase_type == "stemmed":
        recurrent = False
        batch_size = 218
        per_epoch = 2048
    elif model_type == "ann" and phrase_type == "lemmed":
        recurrent = False
        batch_size = 512
        per_epoch = 2048
    else:
        recurrent = True
        batch_size = 64

    if phrase_type == "norm":
        input_dim = 15226
    elif phrase_type == "stemmed":
        input_dim = 10989
    else:
        input_dim = 13916

    training_generator = DataGenerator(list_IDs=partition["train"],
                                       labels=labels,
                                       phrase_type=phrase_type,
                                       batch_size=batch_size,
                                       dim=input_dim,
                                       recurrent=recurrent)
    validation_generator = DataGenerator(list_IDs=partition["validation"],
                                         labels=labels,
                                         phrase_type=phrase_type,
                                         batch_size=batch_size,
                                         dim=input_dim,
                                         recurrent=recurrent)

    # get model
    models = {"ann": Dense(32, input_dim=input_dim, activation="relu"),
              "rnn": SimpleRNN(32, input_dim=input_dim),
              "lstm": LSTM(32, input_dim=input_dim),
              "gru": GRU(32, input_dim=input_dim)}
    model_core = models[model_type]

    # create model
    model = Sequential()
    model.add(model_core)
    model.add(Dense(5, activation="softmax"))

    # compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    model.summary()

    # create callbacks
    # - save model and best weights
    saving_model_weights = ModelCheckpoint("C:\\Users\\15713\\Desktop\\DS Projects\\Sentiment Analysis on Movie "
                                           "Reviews\\sentiment-analysis-on-movie-reviews\\models\\" + phrase_type +
                                           "_" + model_type + "_model.hdf5", monitor="val_acc", verbose=0,
                                           save_best_only=True, save_weights_only=False, mode="auto", period=10)
    # - reduce learning rate when close to optimum
    reduce_lr = ReduceLROnPlateau(monitor="val_acc", factor=0.1, patience=20, verbose=0, mode="auto",
                                  min_delta=0.0001, min_lr=0)
    # - if NaN occurs, stop model
    nan_problem = TerminateOnNaN()
    # - stop training if validation accuracy is not changing or getting worse
    early_stop = EarlyStopping(monitor="val_acc", min_delta=0, patience=20, verbose=0, mode="auto", baseline=None,
                               restore_best_weights=True)
    callbacks_list = [early_stop, nan_problem, reduce_lr, saving_model_weights]

    # fit model
    # - called differently if using batch_generator (check out the source)
    history = model.fit_generator(generator=training_generator,
                                  epochs=500,
                                  steps_per_epoch=99875 // per_epoch,
                                  validation_data=validation_generator,
                                  validation_steps=24969 // per_epoch,
                                  callbacks=callbacks_list)

    # evaluate model
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training Accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
    plt.legend()
    plt.title("Training Accuracy Vs. Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("C:\\Users\\15713\\Desktop\\DS Projects\\Sentiment Analysis on Movie "
                "Reviews\\sentiment-analysis-on-movie-reviews\\models\\" + phrase_type + "_" + model_type +
                "_model.png")

    # return
    return model


def plot_confusion_matrix(real_labels,
                          predicted_labels,
                          target_names,
                          title="Confusion Matrix"):
    """
    Plot confusion matrix

    Arguments
    ---------
    :param real_labels:         real labels

    :param predicted_labels:    predicted labels

    :param target_names:        given classification classes such as [0, 1, 2]
                                the class names, for example: ["bed", "dog", "wow"]

    Optional Arguments
    ------------------
    :param title:               the text to display at the top of the confusion matrix plot

    Citation
    --------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Source
    ------
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """

    # get confusion matrix info
    cm = confusion_matrix(real_labels, predicted_labels)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    accuracy = np.trace(cm) / float(np.sum(cm))

    # create confusion matrix plot
    # - plot confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.get_cmap("Blues"))
    # - show color bar
    fig.colorbar(cax)
    # - set and show x and y tick labels
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.set_yticklabels([""] + target_names)
    ax.set_xticklabels([""] + target_names)
    # - rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # - set minor ticks
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    # - draw grid lines based on minor ticks
    ax.grid(which="minor", color="k", linestyle="-", linewidth=2)
    # - hide other grid lines
    plt.grid(False)
    # - show plot, title, and labels
    plt.tight_layout()
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label\naccuracy={:0.4f}; inaccuracy={:0.4f}".format(accuracy, 1 - accuracy))
    plt.show()


def evaluate_model(model_at_hand, model_in_file, X_test, y_test, test_df, phrase_type, is_dl=False, is_recurrent=False):
    """
    Evaluate model

    Arguments
    ---------
    :param model_at_hand:       model readily available to evaluate
                                value is either a model or None

    :param model_in_file:       model saved in file to evaluate
                                can choose ["logistic", "decision_tree", "random_forest",
                                            "ann", "rnn", "lstm", "gru",
                                            None]
                                use either model_at_hand or model_in_file

    :param X_test:              phrases to test on (X in test data)

    :param y_test:              sentiments to test on (y in test data)

    :param test_df:             test data frame

    :param phrase_type:         name of processed phrases used
                                can choose ["norm", "stemmed", "lemmed"]

    Optional Arguments
    ------------------
    :param is_dl:               is using a deep learning model (ann, rnn, lstm, gru)?
                                only needed if model_at_hand is not None

    :param is_recurrent:        is using a recurrent model (rnn, lstm, gru)?
                                only needed if model_at_hand is not None
    """

    # create target names
    target_names = ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"]

    # vectorize phrases
    vectorized_X_test = vectorize_phrases(phrases=X_test, phrase_type=phrase_type)

    # format test phrases
    if model_at_hand is not None:
        if is_dl:
            vectorized_X_test = vectorized_X_test.toarray()
    else:
        if model_in_file == "ann" or model_in_file == "rnn" or model_in_file == "lstm" or model_in_file == "gru":
            vectorized_X_test = vectorized_X_test.toarray()

    if model_at_hand is not None:
        if is_recurrent:
            vectorized_X_test = np.reshape(vectorized_X_test,
                                           (vectorized_X_test.shape[0], 1, vectorized_X_test.shape[1]))
    else:
        if model_in_file == "rnn" or model_in_file == "lstm" or model_in_file == "gru":
            vectorized_X_test = np.reshape(vectorized_X_test,
                                           (vectorized_X_test.shape[0], 1, vectorized_X_test.shape[1]))

    # get and run model
    if model_at_hand is not None:
        model = model_at_hand
        predicted = model.predict(vectorized_X_test).argmax(axis=1)
    else:
        if model_in_file is not None:
            if model_in_file == "logistic" or model_in_file == "decision_tree" or model_in_file == "random_forest":
                saved_model = open(
                    "C:\\Users\\15713\\Desktop\\DS Projects\\Sentiment Analysis on Movie "
                    "Reviews\\sentiment-analysis-on-movie-reviews\\models\\" + phrase_type + "_" + model_in_file +
                    "_model.pickle", "rb")
                model = pickle.load(saved_model)
                saved_model.close()
            else:
                model = load_model("C:\\Users\\15713\\Desktop\\DS Projects\\Sentiment Analysis on Movie "
                                   "Reviews\\sentiment-analysis-on-movie-reviews\\models\\" + phrase_type + "_" +
                                   model_in_file + "_model.hdf5")
            predicted = model.predict(vectorized_X_test).argmax(axis=1)

    # evaluate model
    print("Classification Report")
    print("---------------------")
    report = classification_report(y_test, predicted, target_names=target_names)
    print(report)
    print()

    print("Plot Confusion Matrix")
    print("---------------------")
    plot_confusion_matrix(y_test, predicted, target_names=target_names)
    print()

    print("Wrongly Classified")
    print("------------------")
    indices = [i for i in range(len(y_test)) if y_test[i] != predicted[i]]
    if phrase_type == "norm":
        wrong_predictions = test_df.iloc[indices, [0, 2, 1]]
    elif phrase_type == "stemmed":
        wrong_predictions = test_df.iloc[indices, [0, 3, 1]]
    else:
        wrong_predictions = test_df.iloc[indices, [0, 4, 1]]
    wrong_predictions["Predicted"] = predicted[indices]
    wrong_predictions.reset_index(inplace=True, drop=True)
    display(wrong_predictions)      # display a data frame in a jupyter notebook
