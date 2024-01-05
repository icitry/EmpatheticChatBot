import os


class Constants:
    NLTK_PATH = f'{os.getcwd()}/nltk'

    DATASETS = [
        {
            'path': os.path.join(os.getcwd(), 'dataset', 'emotions.csv'),
            'delimiter': ';'
        },
        {
            'path': os.path.join(os.getcwd(), 'dataset', 'tweet_emotions.csv'),
            'delimiter': ','
        }
    ]

    DATASET_X_COL = 'content'
    DATASET_Y_COL = 'emotion'
    DATASET_COLUMNS = [DATASET_X_COL, DATASET_Y_COL]

    EMOTIONS = ['happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']

    TRAINING_PERCENT = 0.8

    PARSED_DATASET_PATH = os.path.join(os.getcwd(), 'parsed_data', 'data.txt')
    MODEL_PATH = os.path.join(os.getcwd(), 'model', 'stack_model.pkl')
