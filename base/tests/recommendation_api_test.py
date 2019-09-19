import unittest
import santander_kaggle.internal_api as api
import json
class RecommendationApiTest(unittest.TestCase):
    def test_request(self):
        #Simply check for no errors
        req = api.RecommendationRequest(1)
        resp = api.handler.handle_request(req)

    def test_json_request(self):
        #Simply check for no errors
        json_req = {'ncodpers':1}
        resp = api.handler.handle_json_request(json_req)

if __name__ == '__main__':
    unittest.main()