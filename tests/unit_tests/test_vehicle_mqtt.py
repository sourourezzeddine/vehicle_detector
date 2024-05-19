"""! @brief Unit test for vehicle_mqtt module"""
import unittest
from unittest.mock import MagicMock, patch
import json
import sys

try:
    with open('test_config.json') as f:
        main_config = json.load(f)
except :
    raise FileNotFoundError("The file 'test_config.json' was not found.")


sys.path.append(main_config['project_directory'])
from vehicle_mqtt import connect_mqtt, features_to_json, run

class Test_vehicle_mqtt(unittest.TestCase):
    """Test case class for your module."""

    def setUp(self):
        """Set up method to initialize mocked objects."""
        # Mocking the MQTT client
        self.mocked_client = MagicMock()
        self.mocked_client.publish.return_value = None

    def test_connect_mqtt(self):
        """Test for MQTT connection."""
        # Test MQTT connection
        with patch('vehicle_mqtt.mqtt_client.Client', return_value=self.mocked_client):
            client = connect_mqtt()
            self.assertIsNotNone(client)
            client.connect.assert_called_once()

    def test_features_to_json(self):
        """Test for generating JSON message and publishing."""
        # Test generating JSON message and publishing
        features_list = ['class', 'registration', 'brand', 'country', 'color']
        expected_message = {
            "activity": "Monitoring",
            "class": "class",  
            "classificators": [
                {
                    "brand": "brand",
                    "class": "class",  
                    "color": "color",
                    "country": "country",
                    "model": "unable to identify model",
                    "origin": "camera LPM",
                    "registration": "registration",  
                    "uid": "mocked_uuid"  
                }
            ],
            "code": "1002",
            "from": "Public",
            "registration": "registration",  
            "to": "Parc",
            "uidpassage": "mocked_uuid"
        }

        # Mocking uuid generation
        with patch('vehicle_mqtt.uuid.uuid4', return_value='mocked_uuid'):
            with patch('vehicle_mqtt.connect_mqtt', return_value=self.mocked_client):
                features_to_json(features_list)
                self.mocked_client.publish.assert_called_once_with("features_message", json.dumps(expected_message, indent=4))

    def test_run(self):
        """Test for running the module."""
        # Test running the module
        features_list = ['class', 'registration', 'brand', 'country', 'color']
        with patch('vehicle_mqtt.connect_mqtt', return_value=self.mocked_client):
            run(features_list)
            self.mocked_client.loop_start.assert_called_once()

