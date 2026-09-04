import pandas as pd

from sqlalchemy import create_engine    

user ='postgres'
password = '123'
host = '10.0.222.18'
port = '5432'
database = 'tallerdb'

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

sql = 'select * from familia'
familias = pd.read_sql(sql, engine)
print(familias)

sql = 'select * from categoria'
categorias = pd.read_sql(sql, engine)
print(categorias)

vista_cat = categorias.merge(familias, how='inner' , on='idfamilia')
vista_cat

vista_cat_resumen = vista_cat[['categoria', 'familia']]
vista_cat_resumen

vista_cat_resumen.groupby('familia')['categoria'].count()

vista_cat_resumen.pivot_table(index='familia', columns='categoria', aggfunc='size', fill_value = 0)