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

        predict_resp_data = self._get_predict_response(
            query='привет',
            model_id=model_id,
            labels=[]
        )
        labels, probs = self._dicts_to_values_list(predict_resp_data)

        self.assertEqual(
            'hello',
            predict_resp_data[0]['label']
        )

        self.assertEqual(11, len(labels))

        predict_resp_data = self._get_predict_response(
            query='A какая зарплата??',
            model_id=model_id,
            labels=[]
        )
        labels, probs = self._dicts_to_values_list(predict_resp_data)

        self.assertEqual(
            'salary',
            predict_resp_data[0]['label']
        )

        predict_resp_data = self._get_predict_response(
            query='график',
            model_id=model_id,
            labels=[1, 2, 3]
        )
        labels, probs = self._dicts_to_values_list(predict_resp_data)

        self.assertEqual(
            'not_understand',
            predict_resp_data[0]['label']
        )

        self.assertEqual(1, len(labels))

    def test_post_train_request(self):
        train_resp = self.service.post('/train', data=self.request_data)
        train_resp_data = json.loads(train_resp.data)

        self.assertIn('model_id', train_resp_data)
        model_id = train_resp_data['model_id']
        self.test_models_list.append(model_id)

        predict_resp_data = self._get_predict_response(
            query='нет',
            model_id=model_id,
            labels=[]
        )
        labels, probs = self._dicts_to_values_list(predict_resp_data)

        self.assertEqual(
            'no',
            predict_resp_data[0]['label']
        )

        self.assertIn('not_understand', labels)

        predict_resp_data = self._get_predict_response(
            query='',
            model_id=model_id,
            labels=['yes', 'no']
        )
        labels, probs = self._dicts_to_values_list(predict_resp_data)

        self.assertEqual(
            'not_understand',
            predict_resp_data[0]['label']
        )

        self.assertEqual(3, len(labels))

        predict_resp_data = self._get_predict_response(
            query='НЕТ!',
            model_id=model_id,
            labels=['no', 'no', 'yes', 'no']
        )
        labels, probs = self._dicts_to_values_list(predict_resp_data)

        self.assertEqual(
            'no',
            predict_resp_data[0]['label']
        )

        self.assertEqual(3, len(labels))

    def _get_predict_response(self, query, model_id, labels):
        predict_resp = self.service.post(
            '/predict', data={
                'query': query,
                'model_id': model_id,
                'labels': labels
            }
        )
        predict_resp_data = json.loads(predict_resp.data)

        return predict_resp_data

    def _dicts_to_values_list(self, data):
        labels = [_dict['label'] for _dict in data]
        probs = [_dict['proba'] for _dict in data]
        return labels, probs

    def tearDown(self):
        for model_id in self.test_models_list:
            shutil.rmtree(os.path.join(self.models_storage, str(model_id)))
