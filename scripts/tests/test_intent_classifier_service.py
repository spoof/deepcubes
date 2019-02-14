import json
import os
import unittest

import scripts.intent_classifier_service as service
from deepcubes.embedders import LocalEmbedder
from deepcubes.models import LogisticIntentClassifier
from scripts.utils import get_new_model_id


class VeraLiveDialogServiceTest(unittest.TestCase):

    def setUp(self):
        service.app.testing = True
        self.service = service.app.test_client()

        self.model_storage = 'tests/models/intents'
        os.makedirs(self.model_storage, exist_ok=True)

        self.embedder = LocalEmbedder('tests/data/test_embeds.kv')
        self.classifier = LogisticIntentClassifier(self.embedder)

        self.questions, self.answers = [], []

        with open("tests/data/test_dialog.json", "r") as handle:
            data = json.load(handle)

        for label, category in enumerate(data):
            answer = category["answers"][0]

            for question in category["questions"]:
                self.questions.append(question)
                self.answers.append(answer)

        self.classifier.train(self.answers, self.questions, 'lem')
        self.model_id = get_new_model_id(self.model_storage)
        self.clf_params = self.classifier.save()

        self.clf_path = os.path.join(
            self.model_storage, '{}.cube'.format(self.model_id)
        )

        with open(self.clf_path, 'w') as out:
            out.write(json.dumps(self.clf_params))

        self.output_keys = ["answer", "probability",
                            "threshold", "accuracy_score"]

    def test_get_requests(self):
        predict_resp_data = self._get_predict_response(
            query='название',
            model_id=self.model_id,
            resp_type='get'
        )

        self.assertEqual(2, len(predict_resp_data))
        for output in predict_resp_data:
            for key in self.output_keys:
                    self.assertIn(key, output)

    def test_post_train_request(self):

        predict_resp_data = self._get_predict_response(
            query='чем занимается ваша фирма',
            model_id=self.model_id,
            resp_type='post'
        )
        self.assertEqual(2, len(predict_resp_data))
        for output in predict_resp_data:
            for key in self.output_keys:
                    self.assertIn(key, output)

    def _get_predict_response(self, query, model_id, resp_type):
        if resp_type == 'get':
            predict_resp = self.service.get(
                '/predict', data={
                    'query': query,
                    'model_id': model_id,
                }
            )
        else:
            predict_resp = self.service.post(
                '/predict', data={
                    'query': query,
                    'model_id': model_id,
                }
            )
        predict_resp_data = json.loads(predict_resp.data)

        return predict_resp_data

    def tearDown(self):
        os.remove(self.clf_path)
