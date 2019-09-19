"""
Simple implementation of an api, will load predictions from disk then server them
"""

from typing import Iterable
import json
from santander_kaggle.recommendation_data import get_prediction_data


class RecommendationRequest:
    def __init__(self, customer_id: int):
        self.customer_id = customer_id


class RecommendationResponse:
    def __init__(self, products: Iterable[str]):
        self.products = products

    def to_json(self):
        return json.dumps({"added_products": list(self.products)})


class Handler:
    def __init__(self):
        self.prediction_data = get_prediction_data()

    def handle_request(self, request: RecommendationRequest) -> RecommendationResponse:
        # TODO
        prediction = self.prediction_data[
            self.prediction_data["ncodpers"] == request.customer_id
        ]["added_products"]
        return RecommendationResponse(prediction)

    def handle_json_request(self, json_req: dict) -> str:
        internal_req = RecommendationRequest(json_req["ncodpers"])
        res = self.handle_request(internal_req)
        return res.to_json()


handler = Handler()
