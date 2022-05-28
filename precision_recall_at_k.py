from collections import defaultdict
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import KFold
def precision_recall_at_k(predictions, k=5, threshold=4):
    # Đầu tiên ánh xạ các dự đoán cho từng người dùng.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sắp xếp xếp hạng của người dùng theo giá trị ước tính
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Số lượng mục có liên quan
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # Số mặt hàng được đề xuất trong k hàng đầu
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # Số mục có liên quan và được đề xuất trong top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])
        # Precision @ K: Tỷ lệ các mục được đề xuất có liên quan
        # Khi n_rec_k bằng 0, Độ chính xác là không xác định.
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        # Recall @ K: Tỷ lệ các mục có liên quan được đề xuất
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    return precisions, recalls

data = Dataset.load_builtin('ml-100k')
kf = KFold(n_splits=5)
algo = SVD()
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)
    # Độ chính xác và recall sau đó có thể được tính trung bình trên tất cả người dùng
    print(sum(prec for prec in precisions.values()) / len(precisions))
    print(sum(rec for rec in recalls.values()) / len(recalls))
    print()