import requests
import json
import os # Aggiungiamo os per leggere le variabili sicure

def emotion_detector(text_to_analyze):
    if not text_to_analyze or not text_to_analyze.strip():
        return {
            'anger': None, 'disgust': None, 'fear': None, 
            'joy': None, 'sadness': None, 'dominant_emotion': None
        }

    url = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
    
    hf_token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    payload = {"inputs": text_to_analyze}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            return {
                'anger': None, 'disgust': None, 'fear': None, 
                'joy': None, 'sadness': None, 'dominant_emotion': None
            }

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

    except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError):
        return {
            'anger': None, 'disgust': None, 'fear': None, 
            'joy': None, 'sadness': None, 'dominant_emotion': None
        }