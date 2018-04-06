import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description="visualise the data in log file")
parser.add_argument("--save_path", help="path to the experiment results", required=True)
args = parser.parse_args()

file_path = args.save_path
assert(os.path.isfile(file_path))

f = open(file_path, "r")
lines = f.readlines()

header = lines[0].split()[1:]

epoch = []
train_loss = []
train_acc = []
dev_loss = []
dev_acc = []
dev_f1 = []
test_loss = []
test_acc = []
test_f1 = []

for i in range(1, len(lines)):
    line = lines[i].split()
    epoch.append(int(line[0]))
    train_loss.append(float(line[1]))
    train_acc.append(float(line[2]))
    dev_loss.append(float(line[3]))
    dev_acc.append(float(line[4]))
    dev_f1.append(float(line[5]))
    test_loss.append(float(line[6]))
    test_acc.append(float(line[7]))
    test_f1.append(float(line[8]))


save_dir = os.path.dirname(os.path.abspath(file_path)) + "/"

fig1 = plt.figure(1)
plt.plot(epoch, train_loss, label="train")
plt.plot(epoch, dev_loss, label="dev")
plt.plot(epoch, test_loss, label="test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.title("Loss")
plt.legend(loc=0)
fig1.savefig(save_dir + "loss.png", dpi=300)

fig2 = plt.figure(2)
plt.plot(epoch, dev_loss, label="dev_loss")
plt.plot(epoch, dev_f1, label="dev_f1")
plt.plot(epoch, test_f1, label="test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.title("Loss")
plt.legend(loc=0)
fig2.savefig(save_dir + "test_f1.png", dpi=300)

# plt.show()
