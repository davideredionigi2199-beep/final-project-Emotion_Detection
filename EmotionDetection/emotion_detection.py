import requests
import json
import os

def emotion_detector(text_to_analyze):
    if not text_to_analyze or not text_to_analyze.strip():
        return {
            'anger': None, 'disgust': None, 'fear': None, 
            'joy': None, 'sadness': None, 'dominant_emotion': None
        }

    url = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
    
    # CONTROLLO 1: Render ha letto la variabile d'ambiente?
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        # Se manca il token, te lo scrivo in faccia sulla pagina web!
        return {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 
                'dominant_emotion': "ERRORE_1: Variabile HF_TOKEN mancante su Render!"}

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": text_to_analyze}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        # CONTROLLO 2: Hugging Face ci ha bloccato?
        if response.status_code != 200:
            return {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 
                    'dominant_emotion': f"ERRORE_2: API bloccata, codice {response.status_code}"}

        formatted_response = response.json()
        predictions = formatted_response[0]
        
        emotion_scores = {'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0, 'sadness': 0.0}
        
        for item in predictions:
            label = item['label']
            if label in emotion_scores:
                emotion_scores[label] = round(item['score'], 4)
                
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        emotion_scores['dominant_emotion'] = dominant_emotion
        
        return emotion_scores

    except Exception as e:
        # CONTROLLO 3: Qualcosa è crashato in Python
        return {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 
                'dominant_emotion': f"ERRORE_3: Python Crash - {str(e)}"}