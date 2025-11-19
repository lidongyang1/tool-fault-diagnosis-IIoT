# -*- coding: utf-8 -*-
# @Time    : 2020/6/27 8:30
# @Author  : Dongyang Li
# @FileName: VisualContentPopularityPrediction.py
# @Software: PyCharm
# @E-mail: lidongyang@mail.sdu.edu.cn
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
Accuracy_score = [0.7773,0.7836,0.9755,0.9877,0.9978] # ,0.9587,
Precision_score = [0.7793,0.8240,0.9765,0.9877,0.9978]#,0.9590
Recall_score = [0.7773,0.7836, 0.9755,0.9877,0.9978]# ,0.9587
F1_socre=[0.7725,0.7805,  0.9754, 0.9876,  0.9978] # ,0.9588
names = ['MLP','SVM','CNN','IMSN','IMSN-LSTM']
bar_width=0.5
x = range(len(Accuracy_score))

fig = plt.figure()

ax = plt.subplot(1,3,1)

ax.yaxis.get_major_formatter().set_powerlimits((1, 2))
ax.set_xlabel('Methods', fontsize=15,fontweight='bold')
#plt.xticks([0,1,2,3,4],labels=names)
ax.set_ylabel("Accuracy", fontsize=15,fontweight='bold')
for i in range(len(Accuracy_score)):
    ax.bar(x[i], Accuracy_score[i],color=get_colors()[i],align="center",hatch= get_my_hatches()[i],label='Accuracy',width=0.5,lw=1.5,ec='black')
    # plt.bar(x[i] + bar_width, Recall_score[i], color=get_colors()[i],align="center", hatch=get_my_hatches()[i], label='Precision', width=0.5, lw=1.5,
    #         ec='black')
ax.legend(names,fontsize=10,loc = 'upper left')
ax.set_ylim(0.7,1)
ax.set_xticks([])


ax = plt.subplot(1,3,2)
ax.set_xlabel('Methods', fontsize=15,fontweight='bold')
ax.set_ylabel("Recall", fontsize=15,fontweight='bold')
for i in range(len(Recall_score)):
    ax.bar(x[i], Recall_score[i],color=get_colors()[i],hatch= get_my_hatches()[i],label=names[i],width=0.5,lw=1.5,ec='black')
ax.legend(names,fontsize=10,loc = 'upper left')
ax.set_xticks([])
ax.set_ylim(0.7,1)

ax = plt.subplot(1,3,3)
ax.set_xlabel('Methods', fontsize=15,fontweight='bold')
ax.set_ylabel("F1", fontsize=15,fontweight='bold')
for i in range(len(F1_socre)):
    ax.bar(x[i], F1_socre[i],color=get_colors()[i],hatch= get_my_hatches()[i],label=names[i],width=0.5,lw=1.5,ec='black')
ax.legend(names,fontsize=10,loc = 'upper left')
ax.set_xticks([])
ax.set_ylim(0.7,1)
plt.show()