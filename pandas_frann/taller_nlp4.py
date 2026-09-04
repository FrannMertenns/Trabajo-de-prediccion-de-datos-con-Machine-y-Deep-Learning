import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB #Modelo que clasificara el texto

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
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(textos)

model = MultinomialNB()
model.fit(x, etiquetas)

nuevos=["La compra fue excelente y rápida",
"Mala calidad del producto, no funciona",
"No volvería a comprar jamás",
"Estoy muy satisfecho con el servicio"
]

nuevos_vectorizados= vectorizer.transform(nuevos)
print(model.predict(nuevos_vectorizados))
