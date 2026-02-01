from operator import index
import unittest

from regex import P
from src.configs.constants import POS_COL
from src.data.utils import create_s_clusters_dict, find_wp_coefs_inner
import pandas as pd
from pathlib import Path
import numpy as np
from src.configs.constants import POS_COL, WORD_PROPERTY_COLUMNS


# reminder

EYE_METRICS_NICKNAMES = {
    "GD": "IA_FIRST_RUN_DWELL_TIME",
    "FirstFixProg": "IA_FIRST_FIX_PROGRESSIVE",
    "RegPD": "IA_REGRESSION_PATH_DURATION",
    'NF': "IA_FIXATION_COUNT",
    "FirstPassFF": "FirstPassFFD",
    # relevant to us, fixation metrics:
    "FF": "IA_FIRST_FIXATION_DURATION",
    "FP":"IA_FIRST_RUN_DWELL_TIME",
    "TF": "IA_DWELL_TIME",
    "RP":"IA_REGRESSION_PATH_DURATION",
    "SKIP": "IA_SKIP",
}


class TestFeatures(unittest.TestCase):
    """
    Unit tests for create_s_clusters_dict function in src.data.utils
    """


    def setUp(self):
        # Set up initial DataFrame and expected outputs
        src_path = Path.cwd().parents[0]

        predictors = WORD_PROPERTY_COLUMNS        
        np.random.seed(42)

        def rand_linear_function(X, features):
            # expectes X to be an array of word property columns
            betas = np.random.rand(len(features)+1)
            betas = pd.Series(betas, index=["intercept", *X.columns])
            y =  X @ betas.drop("intercept") + betas["intercept"]
            return betas, y
        
        def rand_logit_function(X, features):
            betas = np.random.rand(len(features)+1)
            betas = pd.Series(betas, index=["intercept", *X.columns])

            linear = X @ betas.drop("intercept") + betas["intercept"]

            # sigmoid
            probs = 1 / (1 + np.exp(-linear))

             # deterministic 'y' for testing
            # y = probs
            y = y = pd.Series(np.random.binomial(1, probs), index=X.index)

            # y = pd.Series(y, index=X.index)

            return betas, y

        
        def get_random_X(n_samples, columns)-> pd.DataFrame:
            X = pd.DataFrame(
                np.random.normal(size=(n_samples, len(columns))),  # independent draws
                columns=columns
            )
            return X

        X = get_random_X(100, predictors)
        data = X
        betas = {}
        
        fixation_metrics = ["FF", "FP", "TF", "RP", "SKIP"]
        for fix_met in fixation_metrics:
            if fix_met == "SKIP":
                random_betas, y = rand_logit_function(X, predictors)
            else:
                random_betas, y = rand_linear_function(X, predictors)
            
            y = pd.DataFrame(y, columns=[fix_met])
            data = pd.concat([data, y], axis=1)
            
            betas[f"{fix_met}_intercept"] = random_betas['intercept']

            for wp in WORD_PROPERTY_COLUMNS:
                betas[f"{fix_met}_{wp}_coef"] = random_betas[wp]

        self.data = data
        
        self.betas = betas

    # def test_wp_coefs(self):
    #     pass
    # TODO: maybe add test on normalizing but i thinkk its a. redundent since we know the normalization works in other stuff
    # and b. its difficult to do that here

    def test_wp_coefs_no_norm(self):
        # Test case 2: 'sentence' input with RR
        coefs = find_wp_coefs_inner(self.data, ["FF", "FP", "TF", "RP"], normalize_RS=False)
        coefs_rounded = {key:round(value, 5) for key, value in coefs.items()}
        betas_rounded = {key:round(value, 5) for key, value in self.betas.items() if "SKIP" not in key}

        assert coefs_rounded == betas_rounded, f"Expected:\n {betas_rounded},\n but got \n{coefs_rounded}"

    # def test_wp_coefs_no_norm_skip(self):
    #     coefs = find_wp_coefs_inner(self.data, ["SKIP"], normalize_RS=False)
    #     for key in coefs:
    #         # check that signs match and coefficients are close
    #         assert np.isclose(coefs[key], self.betas[key], atol=0.5), f"{key} not close,{coefs[key]}, {self.betas[key]} "
        # TODO: chdck if there is a way to test skip - logit function. 
        # maybe not relevant, maybe skip isnt part of theses features


if __name__ == '__main__':
    unittest.main()