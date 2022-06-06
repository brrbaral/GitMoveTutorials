from mgovernUtils import *
from mgovernMetrics import *

# PREPROCESS GOVERNANCE: process the submitted data
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


# MODEL COGERNANCE SUMMARY SCORES: 
def modelGovSummary(actual,pred):
  psi=psi_index(actual,pred)*100
  _,_,csi=characteristStability(x_train,x_test)
  actvsexp=chisquare_test(actual,pred)*100
  return psi,csi,actvsexp


# MODEL DRIFT DETAILS:
# (A) MODEL DISCRIMINATION ANALYSIS:

def model_discrimination_scores(actual,pred,regression=0,forecast=0):
	if regression==1 or forecast==1:
		gini_ind=gini(actual,pred)
		normalized_gini=gini_normalized(actual,pred)
		max_gini=gini_max(actual,pred)
		robinHoodIdx=pietra_ratio(actual,pred)
		ks_test=kstest(actual,pred)
		# aucScore=auROCcurve(actual,pred)     
		return gini_ind,normalized_gini,max_gini,robinHoodIdx,ks_test.pvalue,ks_test.statistic
	else:
		gini_ind=gini(actual,pred)
		normalized_gini=gini_normalized(actual,pred)
		max_gini=gini_max(actual,pred)
		robinHoodIdx=pietra_ratio(actual,pred)
		ks_test=kstest(actual,pred)
		aucScore=auROCcurve(actual,pred)     
		return gini_ind,normalized_gini,max_gini,robinHoodIdx,ks_test.pvalue,ks_test.statistic,aucScore


# (B) SYSTEM OR POPULATION STABILITY:

def populationStabilityScores(actual,pred):
  psi=psi_index(actual,pred)*100
  psi_events=psi_index_events(pred)*100
  psi_nonEvents=psi_index_events(actual)*100
  return psi,psi_events,psi_nonEvents


# (C) CHARACTERISTICS STABILITY INDEX(CSI)
def find_csi(x_train,x_test):
	numr_cols,values,csi=characteristStability(x_train,x_test)
	return numr_cols,values,csi


# (D) ACTUAL VS PREDICTED CALIBRATION
def actual_vs_predicted_calib(actual,pred,forecast=0,regression=0):
	if forecast==0 and regression==0:
		chiSquareSc=chisquare_test(actual,pred)*100
		brierScore=brier_score_loss(actual,pred)*100

		#calib_curve.show() will plot the graph in output
		calib_curve=plot_model_reliability(actual,pred)
		return chiSquareSc,brierScore,calib_curve

# (E) RETURN THE PERFORMACE METRICS FOR THE REGRESSION AND FORECAST METRICS
def reg_forc_performance_metrics(actual,pred,forecast=0,regression=0):
	if forecast==0 and regression==1:
		meanPred=mean_of_pred(pred)
		stDeviation=std_of_prediction(pred)
		range=range_of_prediction(pred)
		coeff_of_var=find_coeff_of_variation(pred)
		r2=find_r2_score(pred,actual)
		
		# actual_pred_graph=plot_actual_vs_predicted(actual,pred)
		# actual_pred_graph.show() will plot the graph
		mse=find_meanSq_error(pred,actual)
		rmse=find_root_meanSq_error(pred,actual)
		nrmse=find_nrmse(pred,actual)
		mae=find_meanabsolute_error(pred,actual)
		mape=find_mape(pred,actual)
		smape=find_smape(pred,actual)
		wape=find_wape(pred,actual)

		return meanPred,stDeviation,range,coeff_of_var,r2,mse,rmse,nrmse,mae,mape,smape,wape

	elif forecast==1 and regression==0:
		r2=find_r2_score(pred,actual)
		actual_pred_graph=plot_actual_vs_predicted(actual,pred)
		# actual_pred_graph.show() will plot the graph
		mse=find_meanSq_error(pred,actual)
		rmse=find_root_meanSq_error(pred,actual)
		nrmse=find_nrmse(pred,actual)
		mae=find_meanabsolute_error(pred,actual)
		mape=find_mape(pred,actual)
		smape=find_smape(pred,actual)
		wape=find_wape(pred,actual)
		return r2,actual_pred_graph,mse,rmse,nrmse,mae,mape,smape,wape

print(find_accuracy(actual,pred))