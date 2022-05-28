from surprise import KNNBasic
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import cross_validate


print('')
print('---------------SVD result-------------')
data = Dataset.load_builtin('ml-100k')
algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print('')
print('---------------PMF result--------------')
data = Dataset.load_builtin('ml-100k')
algo = SVD(biased=False)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print('')
print('----------------NMF result--------------')
data = Dataset.load_builtin('ml-100k')
algo = KNNBasic(sim_options = {'user_based':True})
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# print_perf(perf)

print('')
print('User Based Collaborative Filtering algorithm result')
data = Dataset.load_builtin('ml-100k')
algo = KNNBasic(sim_options = {'user_based': False })
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


print('')
print('Item Based Collaborative Filtering algorithm result')
data = Dataset.load_builtin('ml-100k')
algo = KNNBasic(sim_options = {'user_based': False})
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print('')
print('MSD----User Based Collaborative Filtering algorithm result')
algo = KNNBasic(sim_options = {'name':'MSD','user_based': True})
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print('')
print('cosin----User Based Collaborative Filtering algorithm result')
algo = KNNBasic(sim_options = {'name':'cosine','user_based': True})
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print('')
print('Person sim----User Based Collaborative Filtering algorithm result')
algo = KNNBasic(sim_options = {'name':'pearson','user_based': True})
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print('')
print('--Neighboors--User Based Collaborative Filtering algorithm result')
algo = KNNBasic(k=10, sim_options = {'name':'MSD', 'user_based':True })
cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)







