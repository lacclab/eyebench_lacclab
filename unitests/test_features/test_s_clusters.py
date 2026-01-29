from operator import index
import unittest
from src.configs.constants import POS_COL
from src.data.utils import create_s_clusters_dict
import pandas as pd
from pathlib import Path
import numpy as np
from src.configs.constants import POS_COL

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

        columns = ['trial_id', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION', "IA_SKIP", POS_COL]
        data = [
            [1, 150, 300, 500, 200, 0, "ADJ"],
            [2, 160, 320, 520, 220, 1, "ADJ"],
            [3, 170, 340, 540, 240, 0, "VERB"],
            [4, 180, 360, 560, 260, 1, "VERB"],
            [4, 180, 360, 560, 260, 1, "NOUN"],
        ]
        self.input_df = pd.DataFrame(data, columns=columns)
        data_partial = [row[1:-1] for row in data]
        expected_means = np.array(data_partial).mean(axis=0)
        indx = {"FF":0, "FP":1, "TF":2, "RP":3, "SKIP":4}
        # print("Expected means:", {next(k for k, v in indx.items() if v == i)
        #                           :expected_means[i] for i in range (len(expected_means)) })

        expedcted_dict_no_norm = {
            "ADJ_universal_pos_FF": 155.0,
            "ADJ_universal_pos_FP": 310.0,
            "ADJ_universal_pos_TF": 510.0,
            "ADJ_universal_pos_RP": 210.0,
            "ADJ_universal_pos_SKIP": 0.5,
            "VERB_universal_pos_FF": 175.0,
            "VERB_universal_pos_FP": 350.0,
            "VERB_universal_pos_TF": 550.0,
            "VERB_universal_pos_RP": 250.0,
            "VERB_universal_pos_SKIP": 0.5,
            "NOUN_universal_pos_FF": 180.0,
            "NOUN_universal_pos_FP": 360.0,
            "NOUN_universal_pos_TF": 560.0,
            "NOUN_universal_pos_RP": 260.0,
            "NOUN_universal_pos_SKIP": 1.0,
        }
        
        expedcted_dict_norm = {}
        for col in expedcted_dict_no_norm.keys():
            expedcted_dict_norm[col] = float(expedcted_dict_no_norm[col] / expected_means[indx[col.split('_')[-1]]])

        self.expected_dict = expedcted_dict_norm
        self.expected_dict_no_norm = expedcted_dict_no_norm

    def test_s_clusters(self):
        # Test case 2: 'sentence' input with RR
        result_df = create_s_clusters_dict(self.input_df, ["FF", "FP", "TF", "RP", "SKIP"], criterion=POS_COL)
        assert result_df == self.expected_dict, f"Expected:\n {self.expected_dict},\n but got \n{result_df}"

    def test_s_clusters_no_norm(self):
        # Test case 2: 'sentence' input with RR
        result_df = create_s_clusters_dict(self.input_df, ["FF", "FP", "TF", "RP", "SKIP"], criterion=POS_COL, normalize_RS=False)
        assert result_df == self.expected_dict_no_norm, f"Expected:\n {self.expected_dict},\n but got \n{result_df}"


if __name__ == '__main__':
    unittest.main()