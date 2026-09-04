import spacy #Es un modelo linguistico-grma

spacy.cli.download("es_core_news_sm")

#---cargando nuestro modelo de NLP---
nlp= spacy.load('es_core_news_sm')
texto="El instructor Pepe dictara  un curso en SENATI" 
tokens = nlp(texto) 
print("____Resultados TOKENS_____ ")
print([t for t in tokens])
print("____LEMATIZACION _____ ")
# for t in tokens:
#     print(t.lemma)   
print([t.lemma for t in tokens])
print("____ETIQUETADO GRAMATICAL(POS-TAGGING) _____ ")
# for t in tokens:
#         print(t.text, t.tag)    
print([(t.text, t.tag) for t in tokens])        
print("____RECONOCIMIENTO DE IDENTIFICADORES _____ ")
# for t in tokens.ents:
#         print(t)
print([t for t in tokens.ents])            