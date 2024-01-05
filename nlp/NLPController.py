import multiprocessing
import os
import pickle
import warnings

import dask.dataframe as ddf
import nltk
from scipy.stats import uniform
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from defs import Constants
from nlp import InputParser, TextPreprocessor

warnings.filterwarnings("ignore")


class NLPController:
    def __init_nltk(self, nltk_path):
        nltk.download('stopwords')
        nltk.download('wordnet')

        # Verify installation and perform it if not present
        if os.path.isdir(nltk_path):
            nltk.data.path.append(f'{nltk_path}/nltk_data')
        else:
            os.mkdir(nltk_path)
            nltk.download('wordnet', f"{nltk_path}/nltk_data")
            nltk.download('omw-1.4', f"{nltk_path}/nltk_data/")
            nltk.data.path.append(f'{nltk_path}/nltk_data')

    def __init__(self, nltk_path):
        self._model = None
        self._enc = None
        self._model_score = 0.0

        self._testing_set = None
        self._training_set = None

        self.__init_nltk(nltk_path)
        self._input_parser = InputParser()
        self._text_preprocessor = TextPreprocessor()

    def __classify_emotion(self, emotion):
        if emotion in Constants.EMOTIONS:
            return emotion
        if emotion in ['sadness', 'worry']:
            return 'sad'
        if emotion in ['happiness', 'enthusiasm', 'fun', 'joy', 'love']:
            return 'happy'
        if emotion in ['hate']:
            return 'disgust'
        return 'invalid'

    def __reduce_emotions_array(self, df, col):
        df[col].apply(str.lower)
        df[col] = df[col].apply(self.__classify_emotion)
        mask = df[col].isin(Constants.EMOTIONS)
        df = df[mask]
        return df

    def __prepare_dataset(self, csv_paths, csv_columns, training_percent):
        print('Preparing dataset.')

        self._input_parser.read_from_file(csv_paths=csv_paths, csv_columns=csv_columns)
        self._training_set, self._testing_set = self._input_parser.get_organized_data(training_percent)

        self._training_set = self.__reduce_emotions_array(self._training_set, Constants.DATASET_Y_COL)
        self._testing_set = self.__reduce_emotions_array(self._testing_set, Constants.DATASET_Y_COL)

        print('Finished preparing dataset.')

    def __preprocess_dataset_entries(self, x_col, y_col):
        print('Preprocessing dataset.')

        self._text_preprocessor.set_tested_col_tag(Constants.DATASET_X_COL)

        ddf_training_set = ddf.from_pandas(self._training_set, npartitions=multiprocessing.cpu_count())
        self._training_set = ddf_training_set.map_partitions(self._text_preprocessor.preprocess_df,
                                                             meta=self._training_set)
        self._training_set = self._training_set.compute()

        ddf_testing_set = ddf.from_pandas(self._testing_set, npartitions=multiprocessing.cpu_count())
        self._testing_set = ddf_testing_set.map_partitions(self._text_preprocessor.preprocess_df,
                                                           meta=self._testing_set)
        self._testing_set = self._testing_set.compute()

        self._enc = LabelEncoder()
        self._enc.fit(self._training_set[y_col].values)

        self._training_set = {
            'x': self._training_set[x_col].values,
            'y': self._enc.transform(self._training_set[y_col].values),
        }

        self._testing_set = {
            'x': self._testing_set[x_col].values,
            'y': self._enc.transform(self._testing_set[y_col].values),
        }

        if not os.path.exists(os.path.dirname(Constants.PARSED_DATASET_PATH)):
            os.mkdir(os.path.dirname(Constants.PARSED_DATASET_PATH))

        pickle.dump({
            'training': self._training_set,
            'testing': self._testing_set,
            'enc': self._enc
        }, open(Constants.PARSED_DATASET_PATH, 'wb+'))

        print('Finished preprocessing dataset.')

    def identity_tokenizer(self, text):
        return text

    def __train_model_util(self):
        print('Training model.')

        self.__prepare_dataset(
            csv_paths=Constants.DATASETS,
            csv_columns=Constants.DATASET_COLUMNS,
            training_percent=Constants.TRAINING_PERCENT
        )

        self.__preprocess_dataset_entries(
            x_col=Constants.DATASET_X_COL,
            y_col=Constants.DATASET_Y_COL
        )

        self._tfidf = TfidfVectorizer(tokenizer=self.identity_tokenizer, stop_words='english', lowercase=False)
        self._tfidf.fit_transform(self._training_set['x'])

        self._model = Pipeline(
            [
                ('vectorizer', self._tfidf),
                ('model', StackingClassifier(
                    estimators=[
                        ('log', LogisticRegression(verbose=3)),
                        ('svm', SVC(verbose=3))
                    ],
                    final_estimator=LogisticRegression(max_iter=10000, penalty='l1', solver='liblinear', verbose=3),
                ))
            ]
        )

        hyper_params = {
            'model__log__penalty': ['l1', 'l2'],
            'model__log__C': uniform(loc=0, scale=4),
            'model__log__solver': ['liblinear', 'lbfgs'],
            'model__log__max_iter': [5000, 10000],
            'model__svm__C': uniform(loc=0.1, scale=99.8),
            'model__svm__gamma': uniform(loc=0.0001, scale=9.9998),
            'model__svm__kernel': ['rbf', 'poly', 'sigmoid'],
            'model__svm__max_iter': [5000, 10000],
        }

        self._model = RandomizedSearchCV(
            self._model, hyper_params, cv=7, refit=True, verbose=3, n_jobs=-1
        )

        self._model.fit(
            self._training_set['x'],
            self._training_set['y']
        )

        print('Finished training model.')
        print('Scoring model.')

        self._model_score = self._model.score(
            self._testing_set['x'],
            self._testing_set['y']
        )

        print('Finished scoring model.')

    def train_model(self, path=None):
        if path and os.path.isfile(path):
            save_obj = pickle.load(open(path, 'rb'))
            self._model = save_obj['model']
            self._enc = save_obj['enc']
            self._model_score = save_obj['score']
        else:
            self.__train_model_util()
            pickle.dump({
                'model': self._model,
                'enc': self._enc,
                'nlp': self._text_preprocessor.nlp_text,
                'score': self._model_score
            }, open(path, 'wb+'))

    def predict(self, text):
        if self._model is None:
            return None

        prediction = self._model.predict([self._text_preprocessor.nlp_text(text)])
        return self._enc.inverse_transform(prediction)[0]

    @property
    def model_score(self):
        return self._model_score
