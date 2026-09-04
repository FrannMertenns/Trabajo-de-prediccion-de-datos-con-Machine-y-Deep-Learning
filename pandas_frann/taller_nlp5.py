import spacy    
from sklearn.feature_extraction.text import TfidfVectorizer #Convertir txt a numero
from sklearn.naive_bayes import MultinomialNB #Modelo que clasificara el texto
from sklearn.model_selection import train_test_split #Entrenamiento y prueba

nlp= spacy.load('es_core_news_sm') #Carga del modelo linguistico    

def limpiar_lematizar(texto):
    doc= nlp(texto.lower()) #Procesamiento natural
    tokens = [t.lemma_ for t in doc if not t.is_punct ] #limpieza 
    return " ".join(tokens) #Resultado de los tokens


textos=["Me encanta este producto",
"Es horrible, no lo recomiendo",
"Excelente atención al cliente",
"La experiencia fue muy mala",
"Estoy feliz con la compra",
"No me gustó para nada",
"Fue una buena compra",
"Una pésima decisión",  
"El servicio fue excelente",
"No volveré jamás",
"No me gustó nada el servicio",
"Jamás volvería a comprar aquí",
"Nada de lo prometido fue cumplido",
"Muy lento y poco profesional",
"Estoy muy satisfecho con el servicio",
"Fue una experiencia muy agradable"
]
etiquetas=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1
]
textos_limpios=[limpiar_lematizar (t) for t in textos] #limpieza de datos

x_train, x_test, y_train, y_test= train_test_split(textos_limpios,etiquetas,test_size=0.2, random_state=42) 
vectorizador= TfidfVectorizer() #Declaramos vectorizados
x_train_vectorizado= vectorizador.fit_transform(x_train)
x_test_vectorizado= vectorizador.transform(x_test)
model= MultinomialNB() #DECLARAMOS MODELO DE MACHINE LEARNING PARA CLASIFICAR TEXTO 
model.fit(x_train_vectorizado, y_train) #entrenamos con x(el texto ya es número), y las etiquetas (y)
nuevos=["La compra fue excelente y rápida",
"Mala calidad del producto, no funciona",
"No volvería a comprar jamás",
"Estoy muy satisfecho con el servicio"
]
nuevos_limpios=[limpiar_lematizar(t) for t in nuevos] #limpiamos el texto nuevo
nuevos_vectorizados = vectorizador.transform(nuevos_limpios) #convertimos a número el texto limpio 
predicciones= model.predict(nuevos_vectorizados) #realizamos una predicción
print(predicciones)