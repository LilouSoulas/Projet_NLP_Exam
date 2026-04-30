import pandas as pd
import json

file = 'outputs/databases/lda_resultats_complet.csv'
df = pd.read_csv(file)

useful_columns = ['annee', 'departement', 'departement-nom','titulaire-soutien',
                  'titulaire-liste', 'taux_chom', 'cleaned_blocks','full_text'
                  ]

#df = df[useful_columns]

#print(df.columns.tolist())

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
docs = df['full_text'].dropna().tolist()

from spacy.lang.fr.stop_words import STOP_WORDS
vectorizer_model = CountVectorizer(stop_words=list(STOP_WORDS))


umpa_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)

topic_model = BERTopic(
    language="multilingual", 
    vectorizer_model=vectorizer_model,
    umap_model=umpa_model,
    verbose=True,
    calculate_probabilities=True,

)

topics, probs = topic_model.fit_transform(docs)

df['bertopic_id'] = topics
import numpy as np
df['bertopic_probability'] = [json.dumps(p.tolist()) if p is not None else None for p in probs]
topic_model.get_topic_info().to_csv('outputs/topics_info.csv', index=False, encoding='utf-8')
#df.to_csv('outputs/lda_bertopic_resultats_complet.csv', index=False, encoding='utf-8')

import os
import pandas as pd
from bertopic import BERTopic

# topic_model = BERTopic(...)
# topics, probs = topic_model.fit_transform(docs)

output_dir = 'outputs/graphs'
#os.makedirs(output_dir, exist_ok=True) # Crée le dossier s'il n'existe pas

print("Génération et sauvegarde des visualisations BERTopic...")

try:
    fig_distance = topic_model.visualize_topics()
    
    fig_distance.write_html(os.path.join(output_dir, 'bertopic_distance_map.html'))
  
    
    print(" - Carte des distances sauvegardée.")
except Exception as e:
    print(f"Erreur lors de la carte des distances : {e}")

try:
    fig_docs = topic_model.visualize_documents(docs, hide_annotations=True)
    fig_docs.write_html(os.path.join(output_dir, 'bertopic_documents_map.html'))
    
    print(" - Nuage de documents sauvegardé.")
except Exception as e:
    print(f"Erreur lors du nuage de documents : {e}")
  
try:
    fig_hierarchy = topic_model.visualize_hierarchy()
    
    fig_hierarchy.write_html(os.path.join(output_dir, 'bertopic_hierarchy.html'))
    
    print(" - Hiérarchie des topics sauvegardée.")
except Exception as e:
    print(f"Erreur lors de la hiérarchie : {e}")

print(f"\nSauvegarde terminée dans le dossier : {output_dir}")
