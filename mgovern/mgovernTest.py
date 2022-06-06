from mgovernMetrics import *
from mgovern import *
import pytest

def preprocessGovernance(target,train_path,test_path,pred_with_id_path,actual_with_id_path,DEBUG=False):
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        #Know your target Variable
        target = target
        #train
        train = pd.read_csv(train_path)
        x_train = train.loc[:, train.columns != target] 
        y_train = train.loc[:, train.columns == target]
        test = pd.read_csv(test_path)
        x_test = test.loc[:, test.columns != target] 
        y_test = test.loc[:, test.columns == target]
        #pred
        pred = pd.read_csv(pred_with_id_path)

        if len(pred.columns.tolist())>1:
          for i in range(len(pred.columns.tolist())):
            pred.rename(columns={pred.columns[i]:target+str(i)},inplace=True)

        # pred = pred.apply(LabelEncoder().fit_transform)
        # pred.rename(columns={pred.columns[0]: "pred"}, inplace=True)
        #actuals
        actual = pd.read_csv(actual_with_id_path)
        # actual = actual.apply(LabelEncoder().fit_transform)
        all_xs=pd.concat([x_train,x_test],ignore_index=True)
        all_ys=pd.concat([y_train,y_test],ignore_index=True)
        actual.rename(columns={actual.columns[0]: "actual"}, inplace=True)
        return train,x_train,x_test,test,pred,actual,y_train,y_test,all_xs,all_ys

train,x_train,x_test,test,pred,actual,y_train,y_test,all_xs,all_ys=preprocessGovernance('targets',
                                          'Train_Breast_cancer_Wiskonsin.csv',
                                          'Test_Breast_cancer_wiskonsin.csv',
                                          'predscore_breast_cancer_wiskinson.csv',
                                          'Truth_Breast_cancer_Wiskonsin.csv',
                                          )


def test_accuracyValidation():
  acc=find_accuracy(actual,pred)
  assert acc>=0.0 and acc<=1.0
  
def test_giniValidation():
  gini_val=gini(actual,pred)
  gini_norm=gini_normalized(actual, pred)
  gini_mx=gini_max(actual, pred)
  assert gini_val>=0.0 and gini_val<=1.0
  assert gini_norm>=0.0 and gini_norm<=1.0
  assert gini_mx>=0.0 and gini_mx<=1.0
  
  
def test_pietra_ratio():
  pietra_rat=pietra_ratio(actual,pred)
  assert pietra_rat>=0 and pietra_rat<=100
  

def test_ks_test():
  value=kstest(actual,pred)
  assert value.statistic>=0.0 and value.statistic<=1.0
  
  
def test_roc():
  value=auROCcurve(actual,pred)
  assert value>=0.0 and value<=1.0

def test_precision_recall():
  value=plot_precision_recall(actual,pred)
  assert value>=0 and value<=1.0
  
# def test_psi_values():
#   psi_val=calculate_psi(actual,pred)
#   psi_event=calculate_psi_events_nonevents(pred)
#   psi_nonevent=calculate_psi_events_nonevents(actual)
#   assert psi_val<0.1 and psi_event<0.1 and psi_nonevent<0.1, "no change required"

def test_csi_value():
  _,_,value=characteristStability(x_train,x_test)
  assert value>=0.0 and value<=100.0, "Value out of range from 0 to 100 %"
  
def test_chi_square_value():
  score=chisquare_test(actual,pred)
  assert score>=0, "Value cannot be Negative"
  
def test_brier_score():
  score=find_brier_score(actual,pred)
  assert score>=0.0 and score<=1.0,"Score should be in rage 0 to 1"
  
def test_mean_of_pred():
  mean_val=mean_of_pred(pred)
  assert mean_val>=0, "value ranges from 0 to infinity"
  
def test_std_of_prediction():
  std_val=std_of_prediction(pred)
  assert std_val>=0, "Standard Deviation should ranges from 0 to infinity"
  
def test_range_of_prediction():
  _,_,range=range_of_prediction(pred)
  assert range>=0, "The range of prediction should be 0 to Infinity"

def test_coeff_of_variation():
  var=find_coeff_of_variation(pred)
  assert var>=0, "The range of coefficient of variation is 0 to Infinity"
  
def test_r2_score():
  val=find_r2_score(actual,pred)
  assert val>=0 and val<=1, "Coefficient of Determination should be in range 0 to 1"

def test_mean_sq_error():
  mean_sq=find_meanSq_error(actual,pred)
  assert mean_sq>=0, "Mean square error should be greater than or equal to 0"
  
def test_root_mean_sq_error():
  rmse=find_root_meanSq_error(actual,pred)
  assert rmse>=0, "rmse shoule be greater than or equal to zero"

def test_nrmse():
  nrmse=find_nrmse(actual,pred)
  assert nrmse>=0 and nrmse<=1, "NRMSE shoule be in range 0 to 1"
  
def test_mae():
  mae=find_meanabsolute_error(actual,pred)
  assert mae>=0, "Mean Absolute Error should be greater than or equal to 0"
  
def test_mape():
  mape=find_mape(actual,pred)
  assert mape>=0 and mape<=100, "Mape range 0 to 100%"
  
def test_smape():
  smape=find_smape(actual,pred)
  assert smape>=0 and smape<=200, "SMAPE ranges from 0 to 200 %"
  
def test_wape():
  wape=find_wape(actual,pred)
  assert wape>=0 and wape<=1, "Value should be in range of 0 to 1"