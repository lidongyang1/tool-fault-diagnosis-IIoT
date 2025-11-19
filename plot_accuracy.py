import numpy as np
import matplotlib.pyplot as plt
def get_colors():
    tableau = [(31, 119, 180),(135,206,235),(255, 127, 14),(244, 164, 96),(44, 160, 44), (214, 39, 40)]
    for i in range(len(tableau)):
        r, g, b = tableau[i]
        tableau[i] = (r / 255., g / 255., b / 255.)
    return tableau
# colors = ['midnightblue','sandybrown','cyan','goldenrod','maroon']
def get_my_hatches():
    patterns = ('' ,'','','','', '')
    return patterns
Accuracy = [0.7731,0.7836,0.9695,0.9707,0.9802,0.9830]
Precision = [0.7713,0.8240, 0.9699, 0.9719, 0.9807,0.9831 ]
Recall = [0.7731,0.7836, 0.9695, 0.9707, 0.9802, 0.9830]
F1 = [0.7696,0.7805,0.9695 ,0.9708 , 0.9802,0.9829 ]

names = ['MLP','SVM','CNN','LSTM','MSCNet','LSTM-MSCNet']
x = range(len(Accuracy))
plt.xlabel('刀具故障诊断算法', fontsize=15,family='Simhei',)
plt.ylabel("准确率", fontsize=15,family='Simhei',)
for i in range(len(Accuracy)):
    plt.bar(x[i], Accuracy[i],color=get_colors()[i],hatch= get_my_hatches()[i],label=names[i],width=0.5,lw=1.5,ec='black')
plt.legend(names,fontsize=10,loc = 'upper left')
plt.xticks([])
plt.ylim(0.7,1)
plt.show()

names = ['MLP','SVM','CNN','LSTM','MSCNet','LSTM-MSCNet']
x = range(len(Precision))
plt.xlabel('刀具故障诊断算法', fontsize=15,family='Simhei',)
plt.ylabel("精确率", fontsize=15,family='Simhei',)
for i in range(len(Accuracy)):
    plt.bar(x[i], Precision[i],color=get_colors()[i],hatch= get_my_hatches()[i],label=names[i],width=0.5,lw=1.5,ec='black')
plt.legend(names,fontsize=10,loc = 'upper left')
plt.xticks([])
plt.ylim(0.7,1)
plt.show()

names = ['MLP','SVM','CNN','LSTM','MSCNet','LSTM-MSCNet']
x = range(len(Accuracy))
plt.xlabel('刀具故障诊断算法', fontsize=15,family='Simhei',)
plt.ylabel("召回率", fontsize=15,family='Simhei',)
for i in range(len(Accuracy)):
    plt.bar(x[i], Recall[i],color=get_colors()[i],hatch= get_my_hatches()[i],label=names[i],width=0.5,lw=1.5,ec='black')
plt.legend(names,fontsize=10,loc = 'upper left')
plt.xticks([])
plt.ylim(0.7,1)
plt.show()

names = ['MLP','SVM','CNN','LSTM','MSCNet','LSTM-MSCNet']
x = range(len(Accuracy))
plt.xlabel('刀具故障诊断算法', fontsize=15,family='Simhei',)
plt.ylabel("F1分数", fontsize=15,family='Simhei',)
for i in range(len(Accuracy)):
    plt.bar(x[i], F1[i],color=get_colors()[i],hatch= get_my_hatches()[i],label=names[i],width=0.5,lw=1.5,ec='black')
plt.legend(names,fontsize=10,loc = 'upper left')
plt.xticks([])
plt.ylim(0.7,1)
plt.show()