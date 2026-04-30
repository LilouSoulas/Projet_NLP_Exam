import pandas as pd
import requests
import time
import json
import os
from tqdm import tqdm

INPUT_FILE = 'data/brute/archelect_search_1988.csv'
OUTPUT_FILE = 'data/ocr_retrieval/ocr_recovered_1988.json'
TIMEOUT_SEC = 3 
DELAY_BETWEEN = 0.5 

df = pd.read_csv(INPUT_FILE)

recovered_data = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        recovered_data = json.load(f)
    print(f"Reprise du projet : {len(recovered_data)} lignes déjà traitées.")

processed_urls = {item['ocr_url'] for item in recovered_data}

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

try:
    for index, row in tqdm(df.iterrows(), total=len(df)):
        url = row['ocr_url']
        
        if url in processed_urls:
            continue
            
        text_content = None
        try:
            response = requests.get(url, headers=headers, timeout=TIMEOUT_SEC)
            if response.status_code == 200:
                text_content = response.text
        except Exception:
            text_content = None 

        recovered_data.append({
            'original_index': index,
            'ocr_url': url,
            'text': text_content
        })

        if index % 5 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(recovered_data, f, ensure_ascii=False, indent=2)
        
        time.sleep(DELAY_BETWEEN)

except KeyboardInterrupt:
    print("\nInterruption manuelle. Sauvegarde des données acquises...")

finally:
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(recovered_data, f, ensure_ascii=False, indent=2)
    print(f"Terminé. Données sauvegardées dans {OUTPUT_FILE}")
