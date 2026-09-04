from transformers import pipeline

model = pipeline(
    "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis"
)

opiniones = ["La compra fue excelente y rápida",
"Mala calidad del producto, no funciona",
"No volvería a comprar jamás",
"Estoy muy satisfecho con el servicio"
]

resultados = model(opiniones)
print(resultados)
