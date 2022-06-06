from mgovernUtils import *

# FOR MODEL DRIFT DETAILS:

# (A) MODEL DISCRIMINATION ANALYSIS

# ACCURACY
def find_accuracy(actual,pred):
  import pandas as pd
  from sklearn.metrics import accuracy_score

  if len(pred.columns.tolist())==1:
    pred_sing=pred.round()
    accuracy=accuracy_score(actual,pred_sing)
  else:
    pred_sing=pd.DataFrame(pred.idxmax(axis=1),columns=['preds'])
    pred_sing['preds']=pred_sing['preds'].apply(lambda x: pred.columns.get_loc(x))
    accuracy=accuracy_score(actual,pred_sing)
  return accuracy

# find_accuracy(actual,pred)


#1. PLOT LORENZ CURVE: 

def lorenz_curve(actual,pred):
  import numpy as np
  import pandas as pd
  actual=actual[actual.columns[0]]
  if len(pred.columns.tolist())==1:
    pred=pred[pred.columns[0]]
  else: 
    pred_sing=pd.DataFrame(pred.idxmax(axis=1),columns=['preds'])
    pred_sing['preds']=pred_sing['preds'].apply(lambda x: pred.columns.get_loc(x))
    pred=pred_sing[pred_sing.columns[0]]
  data = zip(actual, pred)
  sorted_data = sorted(data, key=lambda d: d[1])
  sorted_actual = [d[0] for d in sorted_data]
  cumulative_actual = np.cumsum(sorted_actual)
  cumulative_index = np.arange(1, len(cumulative_actual)+1)
  cumulative_actual_shares = cumulative_actual / sum(actual)
  cumulative_index_shares = cumulative_index / len(pred)
  # Add (0, 0) to the plot
  x_values = [0] + list(cumulative_index_shares)
  y_values = [0] + list(cumulative_actual_shares)
  # Display the 45° line stacked on top of the y values
  diagonal = [x - y for (x, y) in zip(x_values, y_values)]
  # diag_x=np.linspace(0,1,len(x_values))
  return x_values,y_values,diagonal

# DRAW LORENZ_CURVE
def plot_lorenz_curve(actual,pred):
  import matplotlib.pyplot as plt 
  plt.rcParams["figure.figsize"] = (8,7)
  x_val,y_val,diagonal = lorenz_curve(actual,pred)
  plt.xlim(0,1)
  plt.ylim(0,1)
  plt.stackplot(x_val, y_val, diagonal)
  plt.xlabel('Cumulative Share of Predictions')
  plt.ylabel('Cumulative Share of Actual Values')
  plt.title('Lorenz curve')
  plt.show()

# plot_lorenz_curve(actual,pred)



# 2. Gini coefficient of Accuracy Ratio:
def gini(actual, pred):
  import numpy as np
  import pandas as pd
  if len(pred.columns.tolist())==1:
    pred_sing=pred.copy()
  else:
    pred_sing=pd.DataFrame(pred.idxmax(axis=1),columns=['preds'])
    pred_sing['preds']=pred_sing['preds'].apply(lambda x: pred.columns.get_loc(x))
  
  assert (len(actual) == len(pred_sing))
  all = np.asarray(np.c_[actual, pred_sing, np.arange(len(actual))], dtype=float)
  all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
  totalLosses = all[:, 0].sum()
  giniSum = all[:, 0].cumsum().sum() / totalLosses
  giniSum -= (len(actual) + 1) / 2.
  return giniSum / len(actual)


def gini_normalized(actual, pred):
  return gini(actual, pred) / gini(actual, actual)

# gini_normalized(actual,pred)

# gini_normalized(actual,pred)


def gini_max(actual, pred):
  return gini(actual, actual)

# gini_max(actual,pred)

# 3. Schutz coefficient or Pietra index or Robin Hood Index
def pietra_ratio(actual,pred):
  import numpy as np
  import pandas as pd
  actual = actual[actual.columns[0]] #in case we get + error
  
  if len(pred.columns.tolist())==1:
    pred=pred[pred.columns[0]]
  else: 
    pred_sing=pd.DataFrame(pred.idxmax(axis=1),columns=['preds'])
    pred_sing['preds']=pred_sing['preds'].apply(lambda x: pred.columns.get_loc(x))
    pred=pred_sing[pred_sing.columns[0]]

  # pred = pred['pred']
  data = zip(actual, pred)
  sorted_data = sorted(data, key=lambda d: d[1])
  sorted_actual = [d[0] for d in sorted_data]
  values=sorted_actual
  n = len(values)
  assert(n > 0), 'Empty list of values'
  sortedValues = sorted(values) #Sort smallest to largest
  #Find cumulative totals
  cumm = [0]
  for i in range(n):
    cumm.append(sum(sortedValues[0:(i + 1)]))

  #Calculate Lorenz points
  LorenzPoints = [[], []]
  sumYs = 0           #Some of all y values
  robinHoodIdx = -1   #Robin Hood index max(x_i, y_i)
  for i in range(1, n + 2):
    x = 100.0 * (i - 1)/n
    y = 100.0 * (cumm[i - 1]/float(cumm[n]))
    LorenzPoints[0].append(x)
    LorenzPoints[1].append(y)
    sumYs += y
    maxX_Y = x - y
    if maxX_Y > robinHoodIdx: robinHoodIdx = maxX_Y   
  giniIdx = 1-((100+(100 - (2 * sumYs)/n))/100) #Gini index 

  return robinHoodIdx

# pietra_ratio(actual,pred)

# pietra_ratio(actual,pred)

# 4. REGRAIN TRIGGER -A 



# 5. Kolmogorov–Smirnov (K-S) test
def kstest(actual,pred):
  from scipy.stats import ks_2samp
  import numpy as np
  import pandas as pd

  if len(pred.columns.tolist())==1:
    pred=pred.copy()
  else:
    pred_sing=pd.DataFrame(pred.idxmax(axis=1),columns=['preds'])
    pred_sing['preds']=pred_sing['preds'].apply(lambda x: pred.columns.get_loc(x))
    pred=pred_sing.copy()

  np.random.seed(172431)
  actual=actual.values.reshape(actual.shape[0],)
  pred=pred.values.reshape(pred.shape[0],)
  ks_test = ks_2samp(actual, pred)
  return ks_test

# ks_test=kstest(actual,pred)
# ks_test.statistic,ks_test.pvalue


# 6. Plot ROC  curve:

def draw_roc(actual,pred):
  from sklearn.metrics import roc_curve
  from sklearn.metrics import roc_auc_score
  import matplotlib.pyplot as plt
  plt.style.use('seaborn')
  from sklearn.metrics import roc_curve,auc

  y_test=actual[actual.columns[0]].values

  if len(pred.columns)==1:
    pred_proba=pred[pred.columns[0]].values
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, pred_proba)
    roc_auc = auc(fpr1,tpr1)
    plt.plot(fpr1,tpr1,
            label="ROC curve (area = {0:0.2f})".format(roc_auc),
            color="orange")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.title(' ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('ROC Curve',dpi=300);  
    plt.show()

  else:
    pred_prob = pred.to_numpy()

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh ={}
    roc_auc={}

    n_class = len(pred.columns)

    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(actual[actual.columns[0]].values,
                                              pred_prob[:,i], pos_label=i)
        roc_auc[i]=auc(fpr[i],tpr[i])
    for i in range(n_class):
      plt.plot(fpr[i],tpr[i],
               label="Label{0} vs Others(area = {1:0.2f})".format(str(i),roc_auc[i]))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show() 

# draw_roc(actual,pred)


# 7. Precision-Recall Curve:
def plot_precision_recall(actual,pred):
  # for binary IndexError: index 1 is out of bounds for axis 1 with size 1
  from sklearn.metrics import precision_recall_curve, roc_curve
  from sklearn.preprocessing import label_binarize
  from sklearn.metrics import average_precision_score
  import matplotlib.pyplot as plt
  # plt.style.use('seaborn')

  # %matplotlib inline

  if len((set(actual[actual.columns[0]])))==2:
    precision, recall, _ = precision_recall_curve(actual,pred)
    ap=average_precision_score(actual, pred)
    plt.plot(recall,precision,label='Class:1 AP: {0:0.2f}'.format(ap))

  else:
    Y = label_binarize(actual[[actual.columns[0]]],
                    classes=[i for i in range(len(set(actual[actual.columns[0]])))])
    y_score = pred.to_numpy()

    # precision recall curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range((len(set(actual[actual.columns[0]])))):
        precision[i], recall[i], _ = precision_recall_curve(Y[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=3, label='Label {0} (AP={1:0.2f})'.format(i,average_precision[i]))
  # print(average_precision)
  # print(list(average_precision.values())[-1]) 
    ap = sum(average_precision.values()) / len(average_precision)  
  plt.plot([0, 1], [1, 0], color="navy", lw=1.5, linestyle="--")
  plt.xlabel("recall")
  plt.ylabel("precision")
  plt.legend(loc="best")
  plt.title("precision - recall curve (AP:{0:0.2f})".format(ap))
  plt.show()
  return ap

# plot_precision_recall(actual,pred)

# 8. Confusion-Matrix:
def plot_confusion_matrix(actual,pred): 
  import numpy as np
  import pandas as pd
  from mlxtend.plotting import plot_confusion_matrix
  from sklearn.metrics import confusion_matrix
  # norm_pred=np.round(pred)
  if len(pred.columns.tolist())==1:
    pred_sing=np.round(pred)
    
  else:
    pred_sing=pd.DataFrame(pred.idxmax(axis=1),columns=['preds'])
    pred_sing['preds']=pred_sing['preds'].apply(lambda x: pred.columns.get_loc(x))
  
  mat=confusion_matrix(actual,pred_sing)
  plot_confusion_matrix(conf_mat=mat,figsize=(8,8),show_normed=True)
# plot_confusion_matrix(actual,pred)

# 9. Area under ROC curve:
def auROCcurve(actual,pred):
  from sklearn.metrics import roc_auc_score
  actuals=actual[actual.columns[0]].values
  preds=pred.to_numpy()
  auc_score1 = roc_auc_score(actual,pred,multi_class='ovr')
  return auc_score1
# auROCcurve(actual,pred) 


# PLOT AUROC: 
def draw_AUroc(actual,pred):
  from sklearn.metrics import roc_curve
  from sklearn.metrics import roc_auc_score
  import matplotlib.pyplot as plt
  plt.style.use('seaborn')
  from sklearn.metrics import roc_curve,auc

  y_test=actual[actual.columns[0]].values

  if len(pred.columns)==1:
    pred_proba=pred[pred.columns[0]].values
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, pred_proba)
    roc_auc = auc(fpr1,tpr1)
    plt.plot(fpr1,tpr1,
            label="ROC curve (area = {0:0.2f})".format(roc_auc),
            color="blue",lw=2.2)
    plt.fill_between(fpr1,tpr1,color='skyblue')
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.title(' ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('ROC Curve',dpi=300);  
    plt.show()

  else:
    pred_prob = pred.to_numpy()

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh ={}
    roc_auc={}

    n_class = len(pred.columns)

    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(actual[actual.columns[0]].values,
                                              pred_prob[:,i], pos_label=i)
        roc_auc[i]=auc(fpr[i],tpr[i])
    for i in range(n_class):
      plt.plot(fpr[i],tpr[i],
               label="Label{0} vs Others(area = {1:0.2f})".format(str(i),roc_auc[i]))
      plt.fill_between(fpr[i],tpr[i])
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--")
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show() 

# draw_AUroc(actual,pred)


# (B) SYSTEM OR POPULATION STABILITY:
# 1. Population Stability Index(PSI):
def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    import numpy as np
    import pandas as pd
    if len(expected.columns.tolist())==1:
      expected=expected[expected.columns[0]].to_numpy()
    else:
      pred_sing=pd.DataFrame(expected.idxmax(axis=1),columns=['preds'])
      pred_sing['preds']=pred_sing['preds'].apply(lambda x: expected.columns.get_loc(x))
      expected=pred_sing[pred_sing.columns[0]].to_numpy()

    actual=actual[actual.columns[0]].to_numpy()
    
    def psi(expected_array, actual_array, buckets):
        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))
        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)

# psi index events
def calculate_psi_events_nonevents(expected, buckettype='bins', buckets=10, axis=0):
    import numpy as np
    import pandas as pd
    if len(expected.columns.tolist())==1:
      expected=expected[expected.columns[0]].to_numpy()
    else:
      pred_sing=pd.DataFrame(expected.idxmax(axis=1),columns=['preds'])
      pred_sing['preds']=pred_sing['preds'].apply(lambda x: expected.columns.get_loc(x))
      expected=pred_sing[pred_sing.columns[0]].to_numpy()

    def psi(expected_array, buckets):

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)

        def sub_psi(e_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc) * np.log(e_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i]) for i in range(0, len(expected_percents)))
        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], buckets)

    return(psi_values)


# (C): CHARACTERISTIC STABILITY INDEX (CSI):
def kde_distr(train_var,test_var):
  import pandas as pd
  from sklearn.neighbors import KernelDensity
  import numpy as np
              # X = pd.merge(train_var,test_var)
  X = pd.concat([train_var, test_var], axis=1).reset_index()
  try:
    X.drop(["index"], axis=1, inplace=True)
  except:
    pass
              
  kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
              #does not work with categorical columns(needs some processing)
              #shows result with numerical data

  kde_score = kde.score(X)
              # probX = np.exp(kde_score) 
  return kde_score

def plot_distr(variables,x_train,test): #variables=columns list
  import matplotlib.pyplot as plt
  import seaborn as sns
  count = 1
  totrows = round(len(variables)/2)
  cls_val = []
  for variable in variables :
    count= count +1
    x_val = kde_distr(x_train[variable][:100], test[variable][:100])
    cls_val.append(x_val)
  return cls_val

# INPUT COLUMNS: Returns the numerical columns
def input_columns(training_input):
  import pandas as pd
  import numpy as np
  df = pd.DataFrame(training_input)
  newdf = df.select_dtypes(include=np.number)
  numr_columns = newdf.columns.tolist()
  return numr_columns,len(numr_columns)
# input_columns(train)


def characteristStability(x_train,x_test):
  numr_columns,num_numr_columns= input_columns(x_train)
  numr_columns_test,num_numr_columns_test=input_columns(x_test)

  try:
    if len(numr_columns) > 0:
                #KDEMultivariate
      val = plot_distr(numr_columns,x_train,x_test)
    if len(numr_columns_test) > 0:
      val = plot_distr(numr_columns_test,x_train,x_test)
    else:
      val=0
      csi=0

    gt50=len([x for x in val if abs(x)>50])
    csi=(gt50/len(val))*100
  except:
    val = 0
    csi=0

  return numr_columns,val,csi
# characteristStability(x_train,x_test)


# (D) ACTUAL VS PREDICTED CALIBRATION:

# 1. Chi- Square Test / Hosmer-Lemeshow (HL) TEst
def chisquare_test(actual,pred,regression=0,forecast=0):
  import numpy as np
  import pandas as pd
  from scipy.stats import chisquare,chi2_contingency

  if regression==1 or forecast==1:
    prednp=pred.to_numpy().flatten()
    actualnp=actual.to_numpy().flatten()
    score=np.sum(np.square(prednp-actualnp)/(actualnp+0.001))
  else: #if classification then check binary or multi-class case
    actual_val=actual[actual.columns[0]].value_counts().sort_index()
    if len((set(actual[actual.columns[0]])))==2:
      pred=pred.round()
      pred_val=pred[pred.columns[0]].value_counts().sort_index()
    else:
      pred_sing=pd.DataFrame(pred.idxmax(axis=1),columns=['preds'])
      pred_sing['preds']=pred_sing['preds'].apply(lambda x: pred.columns.get_loc(x))
      pred_val=pred_sing[pred_sing.columns[0]].value_counts().sort_index()

    obs=np.array([pred_val,actual_val])
    score, pval, dof, ex = chi2_contingency(obs)
  return score

# chisquare_test(actual,pred,regression=1)

# 2.  Brier Score Loss:
# brier_score_loss(actual,pred)
def find_brier_score(actual,pred):
  from sklearn.metrics import brier_score_loss
  from sklearn.preprocessing import label_binarize
  if len((set(actual[actual.columns[0]])))==2:
    brier_score=brier_score_loss(actual,pred)
  else:
    Y = label_binarize(actual[[actual.columns[0]]],
                        classes=[i for i in range(len(set(actual[actual.columns[0]])))])
    y_score = pred.to_numpy()
    brier_scores=[]

    for i in range(len(set(actual[actual.columns[0]]))):
      brier_scores.append(brier_score_loss(Y[:,i],y_score[:,i]))
    brier_score=sum(brier_scores)/len(brier_scores)
  return brier_score

# find_brier_score(actual,pred)


# 3. Calibration Curve Shape Test:
def draw_model_calibration_curve(actual,pred):
  from sklearn.preprocessing import label_binarize
  from sklearn.calibration import calibration_curve
  import matplotlib.pyplot as plt
  plt.style.use('seaborn')

  if len((set(actual[actual.columns[0]])))==2:
    x, y = calibration_curve(actual, pred, n_bins = 10, normalize = True)
    plt.plot(y,x,marker='.',label='Classifier Model')
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    plt.title('Calibration Curve')

  else:
    Y = label_binarize(actual[[actual.columns[0]]],
                            classes=[i for i in range(len(set(actual[actual.columns[0]])))])
    Y_prob = pred.to_numpy()

    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    plt.title('Calibration Curve: Multiclass- One vs Rest')
    for i in range(len(set(actual[actual.columns[0]]))):
      x, y = calibration_curve(Y[:,i], Y_prob[:,i], n_bins = 10, normalize = True)
      # Plot calibration curve
      # Plot model's calibration curve
      plt.plot(y, x, marker = '.', label = "Label {}".format(i))
      
  leg = plt.legend(loc = 'upper left')
  plt.xlabel('Average Predicted Probability in each bin')
  plt.ylabel('Ratio of positives')
  plt.show()

# draw_model_calibration_curve(actual,pred)


# RELIABILITY DIAGRAM:
# Ref: https://github.com/hollance/reliability-diagrams
def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    """Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }


def _reliability_diagram_subplot(ax, bin_data, 
                                 draw_ece=True, 
                                 draw_bin_importance=False,
                                 title="Reliability Diagram", 
                                 xlabel="Confidence", 
                                 ylabel="Expected Accuracy"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    """Draws a reliability diagram into a subplot."""
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8*normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1*bin_size + 0.9*bin_size*normalized_counts

    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 240 / 255.
    colors[:, 1] = 60 / 255.
    colors[:, 2] = 60 / 255.
    colors[:, 3] = alphas

    gap_plt = ax.bar(positions, np.abs(accuracies - confidences), 
                     bottom=np.minimum(accuracies, confidences), width=widths,
                     edgecolor=colors, color=colors, linewidth=1, label="Gap")

    acc_plt = ax.bar(positions, 0, bottom=accuracies, width=widths,
                     edgecolor="black", color="black", alpha=1.0, linewidth=3,
                     label="Accuracy")

    ax.set_aspect("equal")
    ax.plot([0,1], [0,1], linestyle = "--", color="gray")
    
    if draw_ece:
        ece = (bin_data["expected_calibration_error"] * 100)
        ax.text(0.98, 0.02, "ECE=%.2f" % ece, color="black", 
                ha="right", va="bottom", transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.set_xticks(bins)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(handles=[gap_plt, acc_plt])


def _confidence_histogram_subplot(ax, bin_data, 
                                  draw_averages=True,
                                  title="Examples per bin", 
                                  xlabel="Confidence",
                                  ylabel="Count"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    """Draws a confidence histogram into a subplot."""
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    ax.bar(positions, counts, width=bin_size * 0.9)
   
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if draw_averages:
        acc_plt = ax.axvline(x=bin_data["avg_accuracy"], ls="solid", lw=3, 
                             c="black", label="Accuracy")
        conf_plt = ax.axvline(x=bin_data["avg_confidence"], ls="dotted", lw=3, 
                              c="#444", label="Avg. confidence")
        ax.legend(handles=[acc_plt, conf_plt])


def _reliability_diagram_combined(bin_data, 
                                  draw_ece, draw_bin_importance, draw_averages, 
                                  title, figsize, dpi, return_fig):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    """Draws a reliability diagram and confidence histogram using the output
    from compute_calibration()."""
    figsize = (figsize[0], figsize[0] * 1.4)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi, 
                           gridspec_kw={"height_ratios": [4, 1]})

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.1)

    _reliability_diagram_subplot(ax[0], bin_data, draw_ece, draw_bin_importance, 
                                 title=title, xlabel="")

    # Draw the confidence histogram upside down.
    orig_counts = bin_data["counts"]
    bin_data["counts"] = -bin_data["counts"]
    _confidence_histogram_subplot(ax[1], bin_data, draw_averages, title="")
    bin_data["counts"] = orig_counts

    # Also negate the ticks for the upside-down histogram.
    new_ticks = np.abs(ax[1].get_yticks()).astype(int)
    ax[1].set_yticklabels(new_ticks)    

    plt.show()

    if return_fig: return fig


def reliability_diagram(true_labels, pred_labels, confidences, num_bins=10,
                        draw_ece=True, draw_bin_importance=False, 
                        draw_averages=True, title="Reliability Diagram", 
                        figsize=(6, 6), dpi=72, return_fig=False):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    """Draws a reliability diagram and confidence histogram in a single plot.
    
    First, the model's predictions are divided up into bins based on their
    confidence scores.

    The reliability diagram shows the gap between average accuracy and average 
    confidence in each bin. These are the red bars.

    The black line is the accuracy, the other end of the bar is the confidence.

    Ideally, there is no gap and the black line is on the dotted diagonal.
    In that case, the model is properly calibrated and we can interpret the
    confidence scores as probabilities.

    The confidence histogram visualizes how many examples are in each bin. 
    This is useful for judging how much each bin contributes to the calibration
    error.

    The confidence histogram also shows the overall accuracy and confidence. 
    The closer these two lines are together, the better the calibration.
    
    The ECE or Expected Calibration Error is a summary statistic that gives the
    difference in expectation between confidence and accuracy. In other words,
    it's a weighted average of the gaps across all bins. A lower ECE is better.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        draw_averages: whether to draw the overall accuracy and confidence in
            the confidence histogram
        title: optional title for the plot
        figsize: setting for matplotlib; height is ignored
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    bin_data = compute_calibration(true_labels, pred_labels, confidences, num_bins)
    return _reliability_diagram_combined(bin_data, draw_ece, draw_bin_importance,
                                         draw_averages, title, figsize=figsize, 
                                         dpi=dpi, return_fig=return_fig)


def reliability_diagrams(results, num_bins=10,
                         draw_ece=True, draw_bin_importance=False, 
                         num_cols=4, dpi=72, return_fig=False):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    """Draws reliability diagrams for one or more models.
    
    Arguments:
        results: dictionary where the key is the model name and the value is
            a dictionary containing the true labels, predicated labels, and
            confidences for this model
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        num_cols: how wide to make the plot
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    ncols = num_cols
    nrows = (len(results) + ncols - 1) // ncols
    figsize = (ncols * 4, nrows * 4)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, 
                           figsize=figsize, dpi=dpi, constrained_layout=True)

    for i, (plot_name, data) in enumerate(results.items()):
        y_true = data["true_labels"]
        y_pred = data["pred_labels"]
        y_conf = data["confidences"]
        
        bin_data = compute_calibration(y_true, y_pred, y_conf, num_bins)
        
        row = i // ncols
        col = i % ncols
        _reliability_diagram_subplot(ax[row, col], bin_data, draw_ece, 
                                     draw_bin_importance, 
                                     title="\n".join(plot_name.split()),
                                     xlabel="Confidence" if row == nrows - 1 else "",
                                     ylabel="Expected Accuracy" if col == 0 else "")

    for i in range(i + 1, nrows * ncols):
        row = i // ncols
        col = i % ncols        
        ax[row, col].axis("off")
        
    plt.show()

    if return_fig: return fig
    
def create_data_for_reliab_and_plot(actual,pred):
  import pandas as pd
  # function to make the confidence of '0' 1-x
  def value_for_confidence(x):
    if x < 0.5:
      return 1-x
    else:
      return x
  
  if len((set(actual[actual.columns[0]])))==2:
    data_for_reliab=pd.concat([actual,pred],axis=1)
    # make the y-pred as 1 for confidence >=0.5, 0<0.5
    data_for_reliab['pred_label']=data_for_reliab['targets'].apply(lambda x: 1 if x>=0.5 else 0)
    # for binary classification we only pass confidence of positive class
    data_for_reliab['targets']=data_for_reliab['targets'].apply(lambda x: value_for_confidence(x))
    y_true = data_for_reliab.actual.values
    y_pred = data_for_reliab.pred_label.values
    y_conf = data_for_reliab.targets.values

  else:
    pred_sing=pd.DataFrame(pred.idxmax(axis=1),columns=['preds'])
    pred_sing['preds']=pred_sing['preds'].apply(lambda x: pred.columns.get_loc(x))
    pred_sing['conf']=pred.max(axis=1)
    reliab=pd.concat([actual,pred_sing],axis=1)
    y_true = reliab.actual.values
    y_pred = reliab.preds.values
    y_conf = reliab.conf.values

  # Override matplotlib default styling.
  import matplotlib.pyplot as plt

  plt.style.use("seaborn")

  plt.rc("font", size=12)
  plt.rc("axes", labelsize=12)
  plt.rc("xtick", labelsize=12)
  plt.rc("ytick", labelsize=12)
  plt.rc("legend", fontsize=12)

  plt.rc("axes", titlesize=16)
  plt.rc("figure", titlesize=16)

  title ="Model Reliability Diagrams"

  fig = reliability_diagram(y_true, y_pred, y_conf, num_bins=10, draw_ece=True,
                            draw_bin_importance="alpha", draw_averages=True,
                            title=title, figsize=(6, 6), dpi=100)
  return fig

# create_data_for_reliab_and_plot(actual,pred)

# PERFORMANCE METRICS FOR REGRESSION AND FORECASTING ALGORITHMS:

# MEAN / MEDIAN OF PREDICTION
def mean_of_pred(pred):
  import numpy as np
  import pandas as pd
  return np.mean(pred.to_numpy())
# mean_of_pred(pred)


# STANDARD DEVIATION OF PREDICTION
def std_of_prediction(pred):
  import numpy as np
  import pandas as pd
  return np.std(pred.to_numpy())
# std_of_prediction(pred)

# RANGE OF PREDICTION
def range_of_prediction(pred):
  import numpy as np
  max=np.max(pred.to_numpy())
  min=np.min(pred.to_numpy())
  range=max-min
  return max,min,range
# range_of_prediction(pred)

# RELATIVE STANDARD DEVIATION / COEFFICIENT OF VARIATION
def find_coeff_of_variation(pred):
  import numpy as np
  import pandas as pd
  mean=np.mean(pred.to_numpy())
  std=np.std(pred.to_numpy())
  var=(std/mean)*100
  return var
# find_coeff_of_variation(pred)



#  R-SQUARED REGRESSION ERROR:
def find_r2_score(pred,actual):
  from sklearn.metrics import r2_score
  score=r2_score(actual,pred)
  return score
# find_r2_score(pred,actual)


#PLOT ACTUAL VS PREDICTED:
def plot_actual_vs_predicted(actual,pred):
  import numpy as np
  import pandas as pd
  import plotly.offline as plot
  import plotly.graph_objs as go
  import plotly.figure_factory as ff

  trace1 = go.Scatter(x=actual.to_numpy().flatten(), y=pred.to_numpy().flatten(),mode='markers'
                            )

  layout = go.Layout(xaxis=dict(title='Actual Values'),
                           yaxis=dict(title=' Predicted Values'),
                           height=500,
                           width=700,
                           )

  fig = go.Figure(data=[trace1], layout=layout)

  fig.update_layout(legend=dict(yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01),
                    title={
                         'text': "Actual vs Predicted Plot",
                         'y':0.9,
                         'x':0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'},)
  fig.show()

# plot_actual_vs_predicted(actual,pred)

# 2 MEAN SQUARED ERRROR:
def find_meanSq_error(pred,actual):
  import numpy as np
  import pandas as pd
  from sklearn.metrics import mean_squared_error
  score=mean_squared_error(actual,pred)
  return score
# find_meanSq_error(pred,actual)

# 3 ROOT MEAN SQUARED ERROR
def find_root_meanSq_error(pred,actual):
  import numpy as np
  import pandas as pd
  import math
  from sklearn.metrics import mean_squared_error
  mean_sq=mean_squared_error(pred,actual)
  rmse=math.sqrt(mean_sq)
  return rmse
# find_root_meanSq_error(pred,actual)

# NORMALIZED ROOT MEAN SQUARED ERROR (NRMSE):
def find_nrmse(pred,actual):
  import numpy as np
  import pandas as pd
  rmse=find_root_meanSq_error(pred,actual)
  nrmse=rmse/(np.max(pred.to_numpy())-np.min(pred.to_numpy()))
  return nrmse
# find_nrmse(pred,actual)


# MEAN ABSOLUTE ERROR (MAE):
def find_meanabsolute_error(pred,actual):
  import numpy as np
  import pandas as pd
  from sklearn.metrics import mean_absolute_error
  score=mean_absolute_error(actual,pred)
  return score
# find_meanabsolute_error(pred,actual)


# 5 MEAN ABSOLUTE PERCENTAGE ERROR:
def find_mape(pred,actual):
  import numpy as np
  import pandas as pd
  preds=pred[pred.columns[0]].to_numpy()
  actuals=actual[actual.columns[0]].to_numpy()

  if 0 in actuals:
    mape = np.mean(np.abs((actuals - preds)/(actuals+0.1)))*100
  else:
    mape = np.mean(np.abs((actuals - preds)/actuals))*100
  return mape
# find_mape(pred,actual)


# SYMMETRIC MEAN ABSOLUTE PERCENTAGE ERROR(SMAPE)
def find_smape(pred,actual):
  import numpy as np
  import pandas as pd
  preds=pred[pred.columns[0]].to_numpy()
  actuals=actual[actual.columns[0]].to_numpy()
  mape = np.mean(np.abs((preds - actuals)/((preds+actuals)/2)))*100
  return mape
# find_smape(actual,pred)


# WEIGHTED ABSOLUTE PERCENTAGE ERROR(WAPE):
def find_wape(pred,actual):
  import numpy as np
  wape=np.sum(np.abs(pred.to_numpy()-actual.to_numpy()))/np.sum(actual.to_numpy())
  return wape
# find_wape(pred,actual)

