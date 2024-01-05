from defs import Constants
from nlp import NLPController


def main():
    nlp_controller = NLPController(nltk_path=Constants.NLTK_PATH)
    nlp_controller.train_model(
        path=Constants.MODEL_PATH
    )
    print(f'Currently trained model has a score of: {nlp_controller.model_score}')


if __name__ == '__main__':
    main()
