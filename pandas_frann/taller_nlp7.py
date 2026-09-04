from transformers import pipeline

pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-es-en"
)

texto = ["La compra fue excelente y rápida"]

resultados = model(texto)
print(resultados)


# from transformers import pipeline

# pipeline(
#     "translation",
#     model2="facebook/nllb-200-distilled-600M",
#     src_lang="spa_Latn",
#     tgt_lang="quy_Latn"
# )


# texto2 = ["La compra fue excelente y rápida"]

# resultados2 = model2(texto2)
# print(resultados2)