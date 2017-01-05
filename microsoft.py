import base64
import json
import requests

def call_vision_api(image_filename, api_keys):
    api_key = api_keys['microsoft']
    post_url = "https://api.projectoxford.ai/vision/v1.0/analyze?visualFeatures=Categories,Tags,Description,Faces,ImageType,Color,Adult&subscription-key=" + api_key

    image_data = open(image_filename, 'rb').read()
    result = requests.post(post_url, data=image_data, headers={'Content-Type': 'application/octet-stream'})
    result.raise_for_status()

    return result.text

def get_standardized_result(api_result):
    output = {
        'tags' : [],
        'captions' : []
    }

    for tag_data in api_result['tags']:
        output['tags'].append((tag_data['name'], tag_data['confidence']))

    for caption in api_result['description']['captions']:
        output['captions'].append((caption['text'], caption['confidence']))

    return output
