import numpy as np

# data = pd.read_excel('FINAL jester 2006-15.xls', header=None)
# data = data.to_numpy()
# np.save('jester.npy', data)

data = np.load('jester.npy')
data = data[:, 1:]

neg_user, neg_item = np.where(data <= 0)
neg_label = np.zeros(len(neg_user), dtype=int)
neg_samples = np.concatenate((neg_user.reshape(-1, 1), neg_item.reshape(-1, 1), neg_label.reshape(-1, 1)), axis=1)

pos_user, pos_item = np.where((data > 0) & (data <= 10))
pos_label = np.ones(len(pos_user), dtype=int)
pos_samples = np.concatenate((pos_user.reshape(-1, 1), pos_item.reshape(-1, 1), pos_label.reshape(-1, 1)), axis=1)

print("length of negative samples:", len(neg_samples))
print("length of positive samples:", len(pos_samples))

samples = np.concatenate((pos_samples, neg_samples), axis=0)
print("length of all samples:", len(samples))

np.savetxt('jester.dat', samples, fmt='%d', delimiter='\t')
