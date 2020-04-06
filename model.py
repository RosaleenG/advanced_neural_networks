from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.reshape(X.shape[0], 28, 28, 1).astype("float32")
        X /= 255

        if y is None:
            return X

        y = np_utils.to_categorical(y, 4)

        return X, y


def keras_builder():
    model = Sequential()
    model.add(
        Conv2D(28, (3, 3), padding="same", input_shape=(28, 28, 1), activation="relu")
    )

    model.add(Conv2D(28, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(56, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(448, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def build_model():
    preprocessor = Preprocessor()

    model = KerasClassifier(build_fn=keras_builder, batch_size=28, epochs=1)

    return Pipeline([("preprocessor", preprocessor), ("model", model)])

