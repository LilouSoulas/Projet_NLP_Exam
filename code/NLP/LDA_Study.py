import pandas as pd

df_lda = pd.read_csv('outputs/lda_resultats_complet.csv', sep=',', encoding='utf-8')

print(df_lda['topic_id'].value_counts())