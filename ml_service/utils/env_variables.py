""" Class definition to get the Environment variables from .env"""
import os
from dotenv import load_dotenv

class Env():

    def __init__(self):
        load_dotenv()
        # Loading the variables from .env file
        # And store it in the private variable in the Env class

        # Scripts related Variables
        self._train_file_path = \
            os.environ.get("step1_file_path_train")
        self._train_cleaned_file_path = \
            os.environ.get("step2_file_path")
        self._train_fengg_file_path = os.environ.get("step3_file_path")
        self._analysis_report_file_path = \
            os.environ.get("step4_file_path")
        self._model_file_path = \
            os.environ.get("step5_file_path")
        self._test_file_path = \
            os.environ.get("step1_file_path_test")
        self._predicted_df_file_path = \
            os.environ.get("step6_file_path")
    
    @property
    def train_file_path(self):
        return self._train_file_path
    
    @property
    def train_cleaned_file_path(self):
        return self._train_cleaned_file_path
    
    @property
    def train_fengg_file_path(self):
        return self._train_fengg_file_path
    
    @property
    def analysis_report_file_path(self):
        return self._analysis_report_file_path


    @property
    def model_file_path(self):
        return self._model_file_path

    @property
    def test_file_path(self):
        return self._test_file_path

    @property
    def predicted_df_file_path(self):
        return self._predicted_df_file_path

if __name__ == '__main__':
    e = Env()
