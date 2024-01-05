import re

import contractions
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from spellchecker import SpellChecker


class TextPreprocessor:
    def __init__(self):
        self._stop_words = stopwords.words('english')
        self._tested_col_tag = None
        self._stemmer = PorterStemmer()

    def __lowercase_text(self, text):
        return str(text).lower()

    def __replace_abbreviation(self, match):
        if match is None:
            return ''
        match = match.group()
        if type(match) in [str, bytes, bytearray]:
            return TextPreprocessorUtil.COMMON_ABBREVIATIONS[match]
        else:
            return ''

    def __transform_abbreviations(self, text):
        abbr_pattern = re.compile(
            r'(?<!\w)(' + '|'.join(
                re.escape(key) for key in TextPreprocessorUtil.COMMON_ABBREVIATIONS.keys()) + r')(?!\w)')
        return [abbr_pattern.sub(lambda x: self.__replace_abbreviation(x), word) for word in text if
                word is not None]

    def __remove_speech_unrelated_terms(self, text):
        text = re.sub(r'<.*?>', '', text)  # html tags
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # urls
        text = re.sub(r'@\S+', '', text)  # tags, mentions
        text = re.sub(r'&\S+', '', text)  # html characters
        text = re.sub(r'[^\x00-\x7f]', '', text)  # non-ascii chars
        return text

    def __remove_punctuation_and_special_chars(self, text):
        text = re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)
        return text

    def __normalize_whitespaces(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text

    def __tokenize(self, text):
        tokens = nltk.word_tokenize(text)

        return list(
            filter(lambda word: word.isalnum(), tokens)
        )

    def __remove_stopwords(self, text):
        filtered = filter(lambda word: word not in self._stop_words, text)

        return list(filtered)

    def __correct_spellings(self, text):
        spell = SpellChecker()
        corrected_text = []
        misspelled_words = spell.unknown(text)

        for word in text:
            if word in misspelled_words:
                corrected_text.append(spell.correction(word))
            else:
                corrected_text.append(word)

        return corrected_text

    def __perform_stemming(self, text):
        text = [self._stemmer.stem(word) for word in text if word is not None]
        return text

    def __fix_contractions(self, text):
        text = contractions.fix(text)
        return text

    def nlp_text(self, text):
        text = self.__lowercase_text(text)
        text = self.__remove_speech_unrelated_terms(text)
        text = self.__remove_punctuation_and_special_chars(text)
        text = self.__fix_contractions(text)
        text = self.__normalize_whitespaces(text)
        text = self.__tokenize(text)
        text = self.__remove_stopwords(text)
        text = self.__transform_abbreviations(text)
        text = self.__correct_spellings(text)
        text = self.__perform_stemming(text)

        text = ' '.join(text)

        return text

    def set_tested_col_tag(self, tag):
        self._tested_col_tag = tag

    def preprocess_df(self, df):
        if self._tested_col_tag is None:
            return df

        df[self._tested_col_tag] = df[self._tested_col_tag] \
            .map(self.__lowercase_text) \
            .map(self.__remove_speech_unrelated_terms) \
            .map(self.__remove_punctuation_and_special_chars) \
            .map(self.__fix_contractions) \
            .map(self.__normalize_whitespaces) \
            .map(self.__tokenize) \
            .map(self.__remove_stopwords) \
            .map(self.__transform_abbreviations) \
            .map(self.__correct_spellings) \
            .map(self.__perform_stemming)

        return df


class TextPreprocessorUtil:
    COMMON_ABBREVIATIONS = {
        "$": " dollar ",
        "€": " euro ",
        "4ao": "for adults only",
        "a.m": "before midday",
        "a3": "anytime anywhere anyplace",
        "aamof": "as a matter of fact",
        "acct": "account",
        "adih": "another day in hell",
        "afaic": "as far as i am concerned",
        "afaict": "as far as i can tell",
        "afaik": "as far as i know",
        "afair": "as far as i remember",
        "afk": "away from keyboard",
        "app": "application",
        "approx": "approximately",
        "apps": "applications",
        "asap": "as soon as possible",
        "asl": "age, sex, location",
        "atk": "at the keyboard",
        "ave.": "avenue",
        "aymm": "are you my mother",
        "ayor": "at your own risk",
        "b&b": "bed and breakfast",
        "b+b": "bed and breakfast",
        "b.c": "before christ",
        "b2b": "business to business",
        "b2c": "business to customer",
        "b4": "before",
        "b4n": "bye for now",
        "b@u": "back at you",
        "bae": "before anyone else",
        "bak": "back at keyboard",
        "bbbg": "bye bye be good",
        "bbc": "british broadcasting corporation",
        "bbias": "be back in a second",
        "bbl": "be back later",
        "bbs": "be back soon",
        "be4": "before",
        "bfn": "bye for now",
        "blvd": "boulevard",
        "bout": "about",
        "brb": "be right back",
        "bros": "brothers",
        "brt": "be right there",
        "bsaaw": "big smile and a wink",
        "btw": "by the way",
        "bwl": "bursting with laughter",
        "c/o": "care of",
        "cet": "central european time",
        "cf": "compare",
        "cia": "central intelligence agency",
        "csl": "can not stop laughing",
        "cu": "see you",
        "cul8r": "see you later",
        "cv": "curriculum vitae",
        "cwot": "complete waste of time",
        "cya": "see you",
        "cyt": "see you tomorrow",
        "dae": "does anyone else",
        "dbmib": "do not bother me i am busy",
        "diy": "do it yourself",
        "dm": "direct message",
        "dwh": "during work hours",
        "e123": "easy as one two three",
        "eet": "eastern european time",
        "eg": "example",
        "embm": "early morning business meeting",
        "encl": "enclosed",
        "encl.": "enclosed",
        "etc": "and so on",
        "faq": "frequently asked questions",
        "fawc": "for anyone who cares",
        "fb": "facebook",
        "fc": "fingers crossed",
        "fig": "figure",
        "fimh": "forever in my heart",
        "ft.": "feet",
        "ft": "featuring",
        "ftl": "for the loss",
        "ftw": "for the win",
        "fwiw": "for what it is worth",
        "fyi": "for your information",
        "g9": "genius",
        "gahoy": "get a hold of yourself",
        "gal": "get a life",
        "gcse": "general certificate of secondary education",
        "gfn": "gone for now",
        "gg": "good game",
        "gl": "good luck",
        "glhf": "good luck have fun",
        "gmt": "greenwich mean time",
        "gmta": "great minds think alike",
        "gn": "good night",
        "g.o.a.t": "greatest of all time",
        "goat": "greatest of all time",
        "goi": "get over it",
        "gps": "global positioning system",
        "gr8": "great",
        "gratz": "congratulations",
        "gyal": "girl",
        "h&c": "hot and cold",
        "hp": "horsepower",
        "hr": "hour",
        "hrh": "his royal highness",
        "ht": "height",
        "ibrb": "i will be right back",
        "ic": "i see", "icq": "i seek you",
        "icymi": "in case you missed it",
        "idc": "i do not care",
        "idgadf": "i do not give a damn fuck",
        "idgaf": "i do not give a fuck",
        "idk": "i do not know",
        "ie": "that is",
        "i.e": "that is",
        "iykyk": "if you know you know",
        "ifyp": "i feel your pain",
        "IG": "instagram",
        "ig": "instagram",
        "iirc": "if i remember correctly",
        "ilu": "i love you",
        "ily": "i love you",
        "imho": "in my humble opinion",
        "imo": "in my opinion",
        "imu": "i miss you",
        "iow": "in other words",
        "irl": "in real life",
        "j4f": "just for fun",
        "jic": "just in case",
        "jk": "just kidding",
        "jsyk": "just so you know",
        "l8r": "later",
        "lb": "pound",
        "lbs": "pounds",
        "ldr": "long distance relationship",
        "lmao": "laugh my ass off",
        "luv": "love",
        "lmfao": "laugh my fucking ass off",
        "lol": "laughing out loud",
        "ltd": "limited",
        "ltns": "long time no see",
        "m8": "mate",
        "mf": "motherfucker",
        "mfs": "motherfuckers",
        "mfw": "my face when",
        "mofo": "motherfucker",
        "mph": "miles per hour", "mr": "mister",
        "mrw": "my reaction when", "ms": "miss",
        "mte": "my thoughts exactly",
        "nagi": "not a good idea",
        "nbc": "national broadcasting company",
        "nbd": "not big deal",
        "nfs": "not for sale",
        "ngl": "not going to lie",
        "nhs": "national health service",
        "nrn": "no reply necessary",
        "nsfl": "not safe for life",
        "nsfw": "not safe for work",
        "nth": "nice to have",
        "nvr": "never",
        "nyc": "new york city",
        "oc": "original content",
        "og": "original",
        "ohp": "overhead projector",
        "oic": "oh i see",
        "omdb": "over my dead body",
        "omg": "oh my god",
        "omw": "on my way",
        "p.a": "per annum",
        "p.m": "after midday",
        "pm": "prime minister",
        "poc": "people of color",
        "pov": "point of view",
        "pp": "pages",
        "ppl": "people",
        "prw": "parents are watching",
        "ps": "postscript",
        "pt": "point",
        "ptb": "please text back",
        "pto": "please turn over",
        "qpsa": "what happens",
        "ratchet": "rude",
        "rbtl": "read between the lines",
        "rlrt": "real life retweet",
        "rofl": "rolling on the floor laughing",
        "roflol": "rolling on the floor laughing out loud",
        "rotflmao": "rolling on the floor laughing my ass off",
        "rt": "retweet",
        "ruok": "are you ok",
        "sfw": "safe for work",
        "sk8": "skate",
        "smh": "shake my head",
        "sq": "square",
        "srsly": "seriously",
        "ssdd": "same stuff different day",
        "tbh": "to be honest",
        "tbs": "tablespooful",
        "tbsp": "tablespooful",
        "tfw": "that feeling when",
        "thks": "thank you",
        "tho": "though",
        "thx": "thank you",
        "tia": "thanks in advance",
        "til": "today i learned",
        "tl;dr": "too long i did not read",
        "tldr": "too long i did not read",
        "tmb": "tweet me back",
        "tntl": "trying not to laugh",
        "ttyl": "talk to you later",
        "u": "you",
        "u2": "you too",
        "u4e": "yours for ever",
        "utc": "coordinated universal time",
        "w/": "with",
        "w/o": "without",
        "w8": "wait",
        "wassup": "what is up",
        "wb": "welcome back",
        "wtf": "what the fuck",
        "wtg": "way to go", "wtpa":
            "where the party at",
        "wuf": "where are you from",
        "wuzup": "what is up",
        "wywh": "wish you were here",
        "yd": "yard",
        "ygtr": "you got that right",
        "ynk": "you never know",
        "zzz": "sleeping bored and tired"
    }
