import requests
import json

def emotion_detector(text_to_analyze):
    """
    Connects to Watson NLP, extracts emotion scores, 
    and returns a formatted dictionary including the dominant emotion.
    """
    # API configuration
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = { "raw_document": { "text": text_to_analyze } }
    
    # Execution of the POST request
    response = requests.post(url, json=myobj, headers=headers)
    
    # Parsing the response into a dictionary
    formatted_response = json.loads(response.text)
    
    # Navigating the JSON structure to extract the emotion scores
    # The scores are located within the first element of 'emotionPredictions'
    emotions = formatted_response['emotionPredictions'][0]['emotion']
    
    # Creating the required output format
    result = {
        'anger': emotions['anger'],
        'disgust': emotions['disgust'],
        'fear': emotions['fear'],
        'joy': emotions['joy'],
        'sadness': emotions['sadness']
    }
    
    # Calculating the dominant emotion (the key with the highest value)
    dominant_emotion = max(result, key=result.get)
    
    # Adding the dominant emotion to the output dictionary
    result['dominant_emotion'] = dominant_emotion
    
    return result