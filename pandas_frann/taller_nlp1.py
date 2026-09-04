#instalar nltk: pip install nltk / py -m pip install nltk
#instalar spacy: pip install spacy
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
nltk.download('punkt_tab')
texto="Hola a todos ¿Cómo estan?, hoy aprenderemos NLP"
print('---Tokenizando con word_tokenize---')
tokens= word_tokenize(texto.lower())
print(tokens)
print('---Tokenizando con TweetTokenizer---')
tokenizer= TweetTokenizer()
tokens= tokenizer.tokenize(texto.lower())
print(tokens)