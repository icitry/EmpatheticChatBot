import os
import pickle

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

from api.model import UserSubmissionDto, BotResponseDto
from nlp import TextPreprocessor

load_dotenv()

GIPHY_API_URL_RANDOM = 'https://api.giphy.com/v1/gifs/random'
GIPHY_API_KEY = os.getenv('GIPHY_API_KEY')

if GIPHY_API_KEY is None:
    print('Did not provide Giphy API key.')
    exit(1)

MODEL_PATH = os.getenv('MODEL_PATH')

if MODEL_PATH is None:
    print('Did not provide trained model path.')
    exit(1)

save_obj = pickle.load(open(MODEL_PATH, 'rb'))
model = save_obj['model']
enc = save_obj['enc']

if model is None or enc is None:
    print('Invalid trained model data.')
    exit(1)

app = FastAPI()

text_preprocessor = TextPreprocessor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_giphy_res(tag):
    params = {
        'api_key': GIPHY_API_KEY,
        'tag': tag,
        'rating': 'g'
    }

    r = requests.get(url=GIPHY_API_URL_RANDOM, params=params)

    if not r.ok:
        return None

    data = r.json()

    return data['data']['embed_url']


def predict(text):
    prediction = model.predict([text_preprocessor.nlp_text(text)])
    return enc.inverse_transform(prediction)[0]


def process_request(message):
    tag = predict(message)
    gif = get_giphy_res(tag)
    return gif


@app.post("/chat")
async def submit_chat_message(user_submission_dto: UserSubmissionDto, response: Response) -> BotResponseDto:
    user_message = user_submission_dto.message
    res = process_request(user_message)

    if res is None:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BotResponseDto(error="Error communicating with Giphy server.")

    return BotResponseDto(data=res)
