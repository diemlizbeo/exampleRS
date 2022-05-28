
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-100k')

# tập hợp mẫu và tập thử nghiệm ngẫu nhiên
# bộ kiểm tra được thực hiện từ 25% xếp hạng.
trainset, testset = train_test_split(data, test_size=.25)

algo = SVD()
# Huấn luyện thuật toán trên tập hợp và dự đoán xếp hạng cho tập kiểm tra
algo.fit(trainset)
predictions = algo.test(testset)

accuracy.rmse(predictions)