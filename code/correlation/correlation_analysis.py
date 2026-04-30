import pandas as pd
import numpy as np

def get_bertopic_sum(prob_list, target_ids):
    if prob_list is None or not isinstance(prob_list, (list, np.ndarray)):
        return 0
    return sum(prob_list[i] for i in target_ids if i < len(prob_list))


file = 'outputs/databases/lda_bertopic_resultats_complet.csv'
df = pd.read_csv(file, dtype={'lda_topic_id': 'int64', 'bertopic_id': 'int64'})

df['lda_mention_chomage'] = np.where(df['lda_topic_id'] == 5, 1, 0)
df = df[df['annee'] == 1993]

df['bertopic_mention_chomage'] = np.where(df['bertopic_id'].isin([3, 9, 12, 14, 15]), 1, 0)

df['lda_score_chomage'] = np.where(df['lda_topic_id'] == 5, df['lda_topic_score'], 0.0)
target_topics = [3, 9, 12, 14, 15]
df['bertopic_score_chomage'] = df['bertopic_probability'].apply(lambda x: get_bertopic_sum(x, target_topics))
print(df['lda_mention_chomage'].value_counts())
print(df['bertopic_mention_chomage'].value_counts())

print(df['titulaire-soutien'].value_counts().head(20))

import numpy as np

def map_political_orientation(df):
    col = df['titulaire-soutien'].str.lower().fillna('')
    
    conditions = [
        col.str.contains('parti socialiste|radicaux de gauche|verts'),
        col.str.contains('communiste|lutte ouvrière|parti ouvrier'),
        col.str.contains('rassemblement pour la république|union pour la démocratie française|nouveaux écologistes'),
        col.str.contains('front national|alliance populaire')
    ]
    
    choices = [
        'Gauche',
        'Extrême Gauche',
        'Droite',
        'Extrême Droite'
    ]
    
    df['orientation'] = np.select(conditions, choices, default='Autre/Inconnu')
    
    df['bloc_clivage'] = np.select(
        [df['orientation'].isin(['Gauche', 'Extrême Gauche']), 
         df['orientation'].isin(['Droite', 'Extrême Droite'])],
        ['Gauche', 'Droite'], 
        default='Autre'
    )
    
    return df

df = map_political_orientation(df)


print(df['orientation'].value_counts())

stats_geo_pol = df.groupby(['departement', 'orientation']).agg({
    'taux_chom': 'mean',
    'lda_intensity_chomage': 'mean',
    'bertopic_intensity_chomage': 'mean'
}).reset_index()

df_xgauche = stats_geo_pol[stats_geo_pol['orientation'] == 'Extrême Gauche']
df_gauche = stats_geo_pol[stats_geo_pol['orientation'] == 'Gauche']
df_droite = stats_geo_pol[stats_geo_pol['orientation'] == 'Droite']
df_xdroite = stats_geo_pol[stats_geo_pol['orientation'] == 'Extrême Droite']

def print_corr(sub_df, label):
    corr_val = sub_df[['taux_chom', 'bertopic_intensity_chomage']].corr().iloc[0, 1]
    print(f"Corrélation {label} (Chômage réel vs Scores BERTopic) : {corr_val:.4f}")

print_corr(df_xgauche, "EXTRÊME GAUCHE")
print_corr(df_gauche, "GAUCHE")
print_corr(df_droite, "DROITE")
print_corr(df_xdroite, "EXTRÊME DROITE")

import seaborn as sns
import matplotlib.pyplot as plt

plot_df = stats_geo_pol.dropna(subset=['taux_chom', 'lda_intensity_chomage'])

sns.set_theme(style="whitegrid")
g = sns.lmplot(
    data=plot_df,
    x='taux_chom', 
    y='lda_intensity_chomage', 
    hue='orientation',    
    palette='viridis',         
    height=6, aspect=1.5,
    scatter_kws={'alpha':0.5}  
)

g.figure.savefig('outputs/graphs/lda_correlation_scatter_line.jpg', dpi=300, bbox_inches='tight')
plt.title("Réactivité du discours au chômage réel par bloc politique (1988)", fontsize=15)
plt.xlabel("Taux de chômage réel (%)", fontsize=12)
plt.ylabel("Intensité de la thématique Chômage (Score BERTopic)", fontsize=12)
plt.show()


