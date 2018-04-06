# coding: utf-8
import matplotlib.pyplot as plt
import csv
import sys
import numpy as np

SHOW_PLOTS = True
SAVE_PLOTS = True

folder = sys.argv[1]
csv_reader = csv.reader(open(folder + "/training.csv", "r"))
header = next(csv_reader)

epoch = []
train_loss = []
train_acc = []
dev_loss = []
dev_acc = []
dev_f1 = []

for row in csv_reader:
    epoch.append(int(row[0]))
    train_loss.append(float(row[1]))
    train_acc.append(float(row[2]))
    dev_loss.append(float(row[3]))
    dev_acc.append(float(row[4]))
    dev_f1.append(float(row[5]))
    
epoch = np.array(epoch)
train_loss = np.array(train_loss)
train_acc = np.array(train_acc)
dev_loss = np.array(dev_loss)
dev_acc = np.array(dev_acc)
dev_f1 = np.array(dev_f1)

f1 = plt.figure(1)
plt.plot(epoch, train_loss, label="train_loss")
plt.plot(epoch, dev_loss, label="dev_loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc=0)
if SAVE_PLOTS:
    f1.savefig(folder + "/1_loss.pdf")

f2 = plt.figure(2)
plt.plot(epoch, train_acc, label="train_acc")
plt.plot(epoch, dev_acc, label="dev_acc")
plt.xlabel("epochs")
plt.ylabel("acc")
plt.legend(loc=0)
if SAVE_PLOTS:
    f2.savefig(folder + "/2_acc.pdf")

f3 = plt.figure(3)
plt.plot(epoch, dev_f1, label="dev_f1")
plt.xlabel("epochs")
plt.ylabel("f1")
plt.legend(loc=0)
if SAVE_PLOTS:
    f3.savefig(folder + "/3_f1.pdf")

if SHOW_PLOTS:
    plt.show()

best_dev_loss_epoch = np.argmin(dev_loss)
print "Best dev loss is: ", dev_loss.min()
print "Best dev loss is at epoch: ", best_dev_loss_epoch+1
print "Dev f1 score at that epoch: ", dev_f1[best_dev_loss_epoch]