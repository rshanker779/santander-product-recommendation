import unittest
from santander_kaggle import recommendation_utilities as utils
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd


class RecommendationUtilitiesTest(unittest.TestCase):
    @property
    def A(self):
        return pd.DataFrame([["A", "B"], ["C", "D"]], columns=["E", "F"])

    def test_encoder_trained(self):
        A = self.A
        one_hot = OneHotEncoder()
        self.assertFalse(utils.is_encoder_trained(one_hot))
        one_hot.fit_transform(A)
        self.assertTrue(utils.is_encoder_trained(one_hot))
        A = A.append(pd.DataFrame([[pd.np.nan, pd.np.nan]], columns=["E", "F"]))
        imputer = SimpleImputer(strategy="most_frequent")
        self.assertFalse(utils.is_encoder_trained(imputer))
        imputer.fit_transform(A)
        self.assertTrue(utils.is_encoder_trained(imputer))

    def test_ordinal_dict(self):
        ordinal = OrdinalEncoder()
        ordinal.fit_transform(self.A)
        ordinal_dict = utils.get_ordinal_dict_from_encoder(ordinal, self.A.columns)
        self.assertEqual(list(ordinal_dict.keys()), list(self.A.columns))


if __name__ == "__main__":
    unittest.main()
