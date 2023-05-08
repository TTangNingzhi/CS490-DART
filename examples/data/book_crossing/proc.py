import pandas as pd

data = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1')
data = data[data['Book-Rating'] != 0]

data['user'] = data.groupby('User-ID').ngroup()
data['item'] = data.groupby('ISBN').ngroup()

print(data['user'].max(), data['item'].max())
# print(data['Book-Rating'].value_counts())
print(len(data[data['Book-Rating'] < 7]), len(data[data['Book-Rating'] >= 7]))

data['rating'] = data['Book-Rating'].apply(lambda x: 1 if x >= 7 else 0)
data[['user', 'item', 'rating']].to_csv('book_crossing.dat', index=False, sep='\t', header=False)
