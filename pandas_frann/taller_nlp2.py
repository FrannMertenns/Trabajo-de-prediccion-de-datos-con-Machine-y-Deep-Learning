
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords #Corpus_> Coleccion de modelos para entrenar una IA #Stopwords_>Es un corpus 
nltk.download('punkt_tab')

#print(stopwords.words('spanish'))
stop_words = stopwords.words('spanish')
texto="Hola a todos ¿Cómo están?, hoy aprenderemos NLP"
tokens= word_tokenize(texto.lower())
texto_limpio= [t for t in tokens if t.isalpha() and not t in stop_words]
print(texto_limpio)

tokenizer= TweetTokenizer()
tokens= tokenizer.tokenize(texto.lower())
texto_limpio= [t for t in tokens if t.isalpha() and not t in stop_words]
print(texto_limpio)