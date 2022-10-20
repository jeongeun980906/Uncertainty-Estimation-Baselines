import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve,roc_auc_score

def plot(id_score, ood_near_score, ood_far_score, name= 'maha',reverse = False):
    data_near = np.concatenate((id_score,ood_near_score))
    data_far = np.concatenate((id_score,ood_far_score))
    t1 = np.zeros_like(id_score)
    t2 = np.ones_like(ood_near_score)
    t3 = np.ones_like(ood_far_score)
    data_label_n = np.concatenate((t1,t2))
    data_label_f = np.concatenate((t1,t3))

    plt.figure(figsize=(10,5))
    plt.title("{}".format(name))
    plt.subplot(1,2,1)
    ncounts, nbins = np.histogram(id_score,bins=40)
    tcounts, tbins = np.histogram(ood_near_score,bins=40)
    fcounts, fbins = np.histogram(ood_far_score,bins=40)

    plt.hist(tbins[:-1], tbins, weights=tcounts, alpha=0.5,label = 'Near OOD')
    plt.hist(nbins[:-1], nbins, weights=ncounts, alpha=0.5, label = 'ID')
    plt.hist(fbins[:-1], fbins, weights=fcounts, alpha=0.5, label = 'Far OOD')
    plt.legend()

    plt.subplot(1,2,2)
    precision, recall, _   = precision_recall_curve(data_label_n,data_near)
    AUROC1 = roc_auc_score(data_label_n,data_near)
    plt.plot(recall, precision, lw=2, c='b', label='near ood')
    precision, recall, _   = precision_recall_curve(data_label_f,data_far)
    AUROC2 = roc_auc_score(data_label_f,data_far)
    plt.plot(recall, precision, lw=2, c='r', label='far ood')
    plt.legend()
    plt.show()
    return AUROC1,AUROC2

def plot_confidence_hist(prob_true,prob_pred,NAME):
    plt.figure()
    plt.title(NAME)
    for e,(t,p) in enumerate(zip(prob_true,prob_pred)):
        if e == len(prob_true)-1:
            width = 1-p
        else:
            width = prob_pred[e+1] - p
        plt.bar(p,t,width,align = 'edge',color=[0,0,1,0.2])
    plt.plot([0,1],[0,1])
    plt.show()
    

class histogram_binning_calibrator:
    def __init__(self,num_bins = 30,min_score = 0, max_score = 1):
        self.num_bins = num_bins
        self.min_score = min_score
        self.max_score = max_score
        self.M = (max_score - min_score)/num_bins
        self.true_ranges = np.arange(0,1,num_bins)

    def fit(self, scores, y_true):
        y_true = np.asarray(y_true)
        true_probs = []
        for i in range(self.num_bins):
            min_score = self.min_score + self.M*(i)
            max_score = self.min_score + self.M*(i+1)
            indxs_1 = (scores>min_score)
            indxs_2  = (scores<max_score)
            indxs = indxs_1 * indxs_2
            true_prob = y_true[indxs]
            bin_size = sum(indxs)
            if sum(indxs)>0:
                true_prob = true_prob.mean()
            else:
                true_prob = 0
            true_probs.append(true_prob)
        self.theta = true_probs

    def inference(self,scores, y_true):
        true_probs = []
        cscore = []
        bin_total = []
        y_true = np.asarray(y_true)
        total = len(y_true)
        for i in range(self.num_bins):
            min_score = self.min_score + self.M*(i)
            max_score = self.min_score + self.M*(i+1)
            indxs_1 = (scores>min_score)
            indxs_2  = (scores<max_score)
            indxs = indxs_1 * indxs_2
            true_prob = y_true[indxs]
            cscore.append(self.theta[i])
            bin_total.append(sum(indxs))
            if sum(indxs)>0:
                true_prob = true_prob.mean()
            else:
                true_prob = 0
            true_probs.append(true_prob)
        cscore = np.asarray(cscore)
        true_probs = np.asarray(true_probs)
        bin_total = np.asarray(bin_total)
        sorted_index = np.argsort(cscore)
        calibrated_score = cscore[sorted_index]
        true_probs = true_probs[sorted_index]
        bin_total = bin_total[sorted_index]
        ece = self.calculate_ece(true_probs,calibrated_score,bin_total, total)
        print("ECE(%): ",ece)
        return calibrated_score, true_probs, ece
    
    def calculate_ece(self,prob_true,prob_pred, bin_total,total):
        prob_pred = np.asarray(prob_pred)
        ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total / total))
        return ece*100