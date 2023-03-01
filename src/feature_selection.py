import pandas as pd
import yaml
from src.utils.feature_selection_utils import (spearman_corr,
                                               convert_result_series_to_df,
                                               chisquare_test,
                                               splitdf_into_Xtrain_n_Ytrain,
                                               mutualinfo_values,
                                               woe_iv_values,
                                               Analysis_Report)
from ml_service.utils.env_variables import Env

e = Env()


with open('.\\src\\utils\\config.yml') as file:
    try:
        config = yaml.safe_load(file)
        print('config read | Completed')
    except yaml.YAMLError:
        print('config read | Error')

traindf_with_feature_engg = pd.read_parquet(e.train_fengg_file_path)
print(traindf_with_feature_engg.head())

res_spear_coef = spearman_corr(traindf_with_feature_engg, 'Y')
spear_coef_df = convert_result_series_to_df(res_spear_coef,
                                            colname_list=['spearman_coefficient'])

lst_score_series = chisquare_test(traindf_with_feature_engg, 'Y')
chi_square_score_df = convert_result_series_to_df(lst_score_series,
                                                  ['chi_square_scores'])

Xtrain, Ytrain = splitdf_into_Xtrain_n_Ytrain(traindf_with_feature_engg, 'Y')

mutual_info_series = mutualinfo_values(Xtrain, Ytrain)
mutual_info_df = convert_result_series_to_df(mutual_info_series,
                                             ['mutual_info_values'])

weight_of_evidence_Info_series = woe_iv_values(Xtrain, Ytrain)
WOE_IV_Sum_df = convert_result_series_to_df(weight_of_evidence_Info_series,
                                            ['WOE_IV_values'])

dfs_list = [spear_coef_df[:-1], chi_square_score_df[:-1],
            mutual_info_df, WOE_IV_Sum_df]
# key = mutual_info_df.index
Analysisdf = Analysis_Report(dfs_list)
Analysisdf.to_parquet(e.analysis_report_file_path, index=False)
