import requests
import json
import os

def emotion_detector(text_to_analyze):
    if not text_to_analyze or not text_to_analyze.strip():
        return {
            'anger': None, 'disgust': None, 'fear': None, 
            'joy': None, 'sadness': None, 'dominant_emotion': None
        }

    # 🚀 NUOVO URL: Modello super-stabile e sempre attivo su Hugging Face
    url = "https://api-inference.huggingface.co/models/SamLowe/roberta-base-go_emotions"
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 
                'dominant_emotion': "ERRORE_1: Variabile HF_TOKEN mancante su Render!"}

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": text_to_analyze}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            return {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 
                    'dominant_emotion': f"ERRORE_2: API bloccata, codice {response.status_code}"}

        # Hugging Face restituisce una lista di liste
        predictions = response.json()[0]
        
        # Filtriamo solo le 5 emozioni che ci servono per il progetto
        emotion_scores = {'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0, 'sadness': 0.0}
        
        for item in predictions:
            label = item['label']
            if label in emotion_scores:
                emotion_scores[label] = round(item['score'], 4)
                
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        emotion_scores['dominant_emotion'] = dominant_emotion
        
        return emotion_scores

    except Exception as e:
        return {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 
                'dominant_emotion': f"ERRORE_3: Python Crash - {str(e)}"}