from collections import defaultdict

from surprise import SVD
from surprise import Dataset


def get_top_n(predictions, n=10):

    # Đầu tiên ánh xạ các dự đoán cho từng người dùng.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Sau đó sắp xếp các dự đoán cho từng người dùng và lấy k người cao nhất.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# Đầu tiên đào tạo một thuật toán SVD trên tập dữ liệu của phim.
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Sau đó dự đoán xếp hạng cho tất cả các cặp (u, i) không có trong tập huấn luyện.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
top_n = get_top_n(predictions, n=10)
# In các mục được đề xuất cho mỗi người dùng
for uid, user_ratings in top_n.items():
    print("ID người dùng:", uid,"=> ID phim được rcm là:", [iid for (iid, _) in user_ratings])