import io 
from surprise import KNNBaseline
from surprise import Dataset

def read_item_names():
    # Đọc tệp u.item từ tập dữ liệu 100-k MovieLens và trả về hai ánh xạ để chuyển đổi id thô thành tên phim và tên phim thành id thô.
    file_name ='ml-100k/u.item'
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid

# đào tạo thuật toán để tính toán sự giống nhau giữa các mục
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

# raw id <-> movie name
rid_to_name, name_to_rid = read_item_names()
# Truy xuất id bên trong của phim
film_raw_id = input('Nhap id phim muon de xuat: ')
film_inner_id = algo.trainset.to_inner_iid(film_raw_id)
# Lấy id bên trong của những người hàng xóm gần nhất của phim
film_neighbors = algo.get_neighbors(film_inner_id, k=10)
# Chuyển đổi id bên trong của hàng xóm thành tên.
film_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in film_neighbors)
film_neighbors = (rid_to_name[rid] for rid in film_neighbors)

print('The 10 nearest neighbors of', rid_to_name[film_raw_id],'are:')
for movie in film_neighbors:
    print(movie)