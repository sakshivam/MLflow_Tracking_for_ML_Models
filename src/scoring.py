import pandas as pd
import yaml
from src.utils.scoring_utils import (Fill_missing_Val_Columns,
                                     Drop_Missing_Val_Columns,
                                     replace_values_using_dict,
                                     predictions)
from ml_service.utils.env_variables import Env

e = Env()

with open('.\\src\\utils\\config.yml') as file:
    try:
        config = yaml.safe_load(file)
        print('config read | Completed')
    except yaml.YAMLError:
        print('config read | Error')

testdf = pd.read_parquet(e.test_file_path)
print(testdf.head())

testdf = Fill_missing_Val_Columns(testdf,
                                  config['MISSING_VAL_COLUMNS_TO_FILL'])
testdf = Drop_Missing_Val_Columns(testdf, config['MISSING_VAL_COLUMNS'])

testdf = replace_values_using_dict(testdf, config['DICT_FOR_CLUBBING'])
testdf = replace_values_using_dict(testdf,
                                   config['DICT_TO_GET_ORDINAL_FEATURES'])
result_df = predictions(testdf, 'Y',
                        config['TEN_BEST_FEATURES_OBSERVED_SELECTION'],
                        e.model_file_path)
result_df.to_parquet(e.predicted_df_file_path, index=False)


