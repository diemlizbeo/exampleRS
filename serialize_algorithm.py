import os
from surprise import SVD
from surprise import Dataset
from surprise import dump

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)
# Tính toán các dự đoán của thuật toán 'ban đầu'.
predictions = algo.test(trainset.build_testset())
# Dump thuật toán và tải lại nó.
file_name = os.path.expanduser('~/dump_file')
dump.dump(file_name, algo=algo)
_, loaded_algo = dump.load(file_name)
# đảm bảo rằng thuật toán vẫn như cũ bằng cách kiểm tra các dự đoán.
predictions_loaded_algo = loaded_algo.test(trainset.build_testset())
if predictions == predictions_loaded_algo:
    print('Predictions are the same')