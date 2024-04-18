import json
import random
from paho.mqtt import client as mqtt_client
import uuid

try:
    with open('MQTT_config.json') as f:
        config = json.load(f)
except :
    raise FileNotFoundError("The file 'MQTT_config.json' was not found.")

# Global Constants for the MQTT part
broker = config['broker_address']
port = config['port']
topic = config['topic']
# Generate a Client ID with the publish prefix.
client_id = f'publish-{random.randint(0, 1000)}'

def connect_mqtt():
    """! connects to MQTT"""
    client = mqtt_client.Client(client_id)
    client.connect(broker, port)
    return client

def features_to_json(features_list):
    """! organizes the identified features of the vehicle into a json message.

    @param file_path path to the video or camera.

    @return a json message.
    """   
    values_list = features_list
    uid = str(uuid.uuid4()) #! Generating a random UUID for uid
    
    json_data = {
        "activity": "Monitoring",
        "class": values_list[0],  
        "classificators": [
            {
                "brand": values_list[2],
                "class": values_list[0],  
                "color": values_list[4],
                "country": values_list[3],
                "model": "unable to identify model",
                "origin": "camera LPM",
                "registration": values_list[1],  
                "uid": uid  
            }
        ],
        "code": "1002",
        "from": "Public",
        "registration": values_list[1],  
        "to": "Parc",
        "uidpassage": uid
    }

    # Convert the dictionary to a JSON string
    json_message = json.dumps(json_data, indent=4)
    connect_mqtt().publish("features_message", json_message)

def run(features_list):
    """! sends the json message using MQTT"""
    client = connect_mqtt()
    client.loop_start()
    features_to_json(features_list) 