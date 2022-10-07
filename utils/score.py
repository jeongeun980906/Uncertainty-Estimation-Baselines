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