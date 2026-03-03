import json
import pandas as pd
import re
import string
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import num2words

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class WowheadPreprocessor:
    def __init__(self, use_stemming=False, use_lemmatization=False, remove_diacritics=True):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_diacritics = remove_diacritics

    def clean_text(self, text):
        if not text or not isinstance(text, str):
            return ""

        text = text.replace('\\n', ' ').replace('\n', ' ')
        text = re.sub(r'<[^>]+>', '', text)

        if self.remove_diacritics:
            text = unidecode(text)

        tokens = re.findall(r'(?:https?://\S+)|(?:\w+(?:[-]\w+)*)|(?:\d+)', text.lower())

        cleaned_tokens = []
        for token in tokens:
            token = token.strip(string.punctuation)
            if not token or token in self.stop_words or len(token) < 2:
                continue

            if self.use_lemmatization:
                token = self.lemmatizer.lemmatize(token)
            elif self.use_stemming:
                token = self.stemmer.stem(token)

            if token.isdigit():
                token = num2words.num2words(token)

            cleaned_tokens.append(token)

        return " ".join(cleaned_tokens)


if __name__ == "__main__":
    input_file = 'wowhead_articles.jsonl'
    data = []

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Filne {input_file} not found.")
        data = []

    if data:
        df = pd.DataFrame(data)

        methods = {
            'lemmatized': WowheadPreprocessor(use_lemmatization=True),
            'stemmed': WowheadPreprocessor(use_stemming=True),
            'basic': WowheadPreprocessor(remove_diacritics=False)
        }

        fields = ['title', 'author', 'content']

        for method_name, processor in methods.items():
            filename = f"tokens_{method_name}.txt"
            print(f"Generating file: {filename}...")

            with open(filename, 'w', encoding='utf-8') as f:
                for field in fields:
                    if field in df.columns:
                        counts = Counter()
                        for text in df[field].dropna():
                            cleaned = processor.clean_text(text)
                            counts.update(cleaned.split())

                        for word, count in counts.most_common():
                            f.write(f"{word} : {count}\n")

                        f.write("\n\n")