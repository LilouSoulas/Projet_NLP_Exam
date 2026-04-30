import pandas as pd
import json

df_orig = pd.read_csv('data/brute/archelect_search_1988.csv')

with open('data/ocr_retrieval/ocr_recovered_1988.json', 'r', encoding='utf-8') as f:
    data_ocr = json.load(f)

df_ocr = pd.DataFrame(data_ocr)

df_final = pd.merge(
    df_orig, 
    df_ocr[['ocr_url', 'text']], 
    on='ocr_url', 
    how='left'
)

nb_succes = df_final['text'].notna().sum()
print(f"Mapping terminé : {nb_succes} textes intégrés sur {len(df_final)} lignes.")

df_final.to_csv('data/final/final_dirty/archelect_1988_complet.csv', index=False, encoding='utf-8')
