import unittest
import shutil
import json
import os

import scripts.vera_live_dialog_service as service


class ServiceTest(unittest.TestCase):

    def setUp(self):
        service.app.testing = True
        self.service = service.app.test_client()

        self.models_storage = 'scripts/models'
        self.test_config_path = 'tests/data/vera_test.config'

        with open(self.test_config_path, 'r') as conf_file:
            self.test_config = conf_file.read()
        self.request_data = {
            'config':  self.test_config
        }
        self.test_models_list = list()

    def test_get_requests(self):
        train_resp = self.service.get('/train', data=self.request_data)
        train_resp_data = json.loads(train_resp.data)

        self.assertIn('model_id', train_resp_data)
        model_id = train_resp_data['model_id']
        self.test_models_list.append(model_id)

        query = 'привет'
        predict_resp = self.service.get(
            '/predict', data={'query': query, 'model_id': model_id}
        )
        predict_resp_data = json.loads(predict_resp.data)
        self.assertEqual(
            'hello',
            predict_resp_data[0]['label']
        )

    def test_post_train_request(self):
        train_resp = self.service.post('/train', data=self.request_data)
        train_resp_data = json.loads(train_resp.data)

        self.assertIn('model_id', train_resp_data)
        model_id = train_resp_data['model_id']
        self.test_models_list.append(model_id)

        query = 'привет'
        predict_resp = self.service.post(
            '/predict', data={'query': query, 'model_id': model_id}
        )
        predict_resp_data = json.loads(predict_resp.data)
        self.assertEqual(
            'hello',
            predict_resp_data[0]['label']
        )

    def tearDown(self):
        for model_id in self.test_models_list:
            shutil.rmtree(os.path.join(self.models_storage, str(model_id)))
