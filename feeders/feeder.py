import numpy as np # numpy, pickle để đọc dữ liệu từ file .npy, .pkl
import pickle
import torch
from torch.utils.data import Dataset # kế thừa lớp dataset của Pytorch để tạo DataLoader
import sys
import random

from graph.sign_27 import Graph # dùng để định nghĩa đồ thị skeleton 27 joints

sys.path.extend(["../"])
import os
from einops import rearrange
from feeders import tools # import các hàm tiền xử lý dữ liệu từ feeders/tools.py

flip_index = np.concatenate( # đây là kĩ thuật mirroring augmentation, tạo ra dữ liệu mới bằng cách đảo ngược dữ liệu cũ cho skeleton
    (
        [0, 2, 1, 4, 3, 6, 5],
        [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ),
    axis=0,
)


class Feeder(Dataset):
    def __init__(
        self,
        data_path, # đường dẫn đến file .npy chứa dữ liệu
        label_path, # đường dẫn đến file .pkl chứa nhãn
        random_choose=False, # chọn ngẫu nhiên một phần của chuỗi đầu vào
        random_shift=False, # dịch chuyển dữ liệu theo thời gian bằng cách thêm hoặc bớt các frame 0 vào đầu hoặc cuối chuỗi
        random_move=False, # áp dụng dịch chuyển dữ liệu theo không gian bằng cách xoay, co giãn, dịch chuyển dữ liệu
        window_size=-1, # giới hạn số khung hình đầu vào
        normalization=False, # chuẩn hóa dữ liệu nếu True
        debug=False, # chế độ debug, để true thì chỉ sử dụng 100 mẫu để kiểm tra
        use_mmap=True, # dùng memory map để load dữ liệu, giúp tiết kiệm bộ nhớ khi đọc dữ liệu lớn
        random_mirror=False,
        random_mirror_p=0.5,
        is_vector=False,
        lap_pe=False,
        bone_stream=False, # kích hoạt bone stream cho multiple-stream fusion
        motion_stream=False, # kích hoạt motion stream cho multiple-stream fusion
        num_class=100 #2000,
    ):
        """

        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param lap_pe: If true, use laplacian positional encoding (only for LapPE attention model)
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.load_data() # gọi hàm load_data() để load dữ liệu và nhãn từ file
        self.is_vector = is_vector
        self.lap_pe = lap_pe
        self.bone_stream = bone_stream
        self.motion_stream = motion_stream
        self.num_class = num_class
        if normalization: # nếu là chuẩn hóa dữ liệu thì gọi hàm get_mean_map() để tính giá trị trung bình và độ lệch chuẩn của dữ liệu
            self.get_mean_map()

        """LapPE giúp tăng cường thông tin không gian của skeleton thông qua đồ thị Laplacian."""
        if self.lap_pe: # nếu là laplacian positional encoding thì tính eig_vals và eig_vecs
            from feeders import posenc
            from torch_geometric.data import Data
            import torch_geometric.transforms as T

            edge_index_per_frame = torch.tensor(Graph().neighbor).long()
            self.edge_index = torch.cat(
                [edge_index_per_frame + i * 27 for i in range(self.window_size)], axis=0
            ).T
            self.num_nodes = self.edge_index.max() + 1

            if os.path.exists(f"data/eig_vals.pt"):
                self.eig_vals = torch.load(f"data/eig_vals.pt")
                self.eig_vecs = torch.load(f"data/eig_vecs.pt")
            else:
                pe_enabled_list = ["LapPE"]
                data = Data(
                    edge_index=self.edge_index,
                    num_nodes=self.num_nodes,
                )
                self.eig_vals, self.eig_vecs = posenc.compute_posenc_stats(
                    data,
                    pe_enabled_list,
                    is_undirected=True,
                    laplacian_norm_type="none",
                    max_freqs=8,
                    eigvec_norm="L2",
                )

                # save eig_vals and eig_vecs
                torch.save(self.eig_vals, f"data/eig_vals.pt")
                torch.save(self.eig_vecs, f"data/eig_vecs.pt")

            self.transform = T.Compose([T.Distance()])

        print(len(self.label))

    """Hàm load_data() dùng để load dữ liệu và nhãn từ file .npy và .pkl"""
    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, "rb") as f:
                self.sample_name, self.label = pickle.load(f, encoding="latin1")

        # load data
        if self.use_mmap: # nếu dùng memory map thì load dữ liệu bằng mmap_mode="r"
            self.data = np.load(self.data_path, mmap_mode="r")
        else:
            self.data = np.load(self.data_path)
        if self.debug: # nếu là chế độ debug thì chỉ sử dụng 100 mẫu
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    """Hàm get_mean_map() dùng để tính giá trị trung bình và độ lệch chuẩn của dữ liệu"""
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = (
            data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        )
        self.std_map = (
            data.transpose((0, 2, 4, 1, 3))
            .reshape((N * T * M, C * V))
            .std(axis=0)
            .reshape((C, 1, V, 1))
        )

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    # 160-163: Đọc một mẫu dữ liệu 
    def __getitem__(self, index):
        data_numpy = self.data[index]  # C T V M
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # nếu dữ liệu có giá trị inf thì gán giá trị 0, dùng cho MLASL
        data_numpy[np.isinf(data_numpy)] = 0  # For MLASL

        # remove null frames
        """index = (data_numpy.sum(-1).sum(-1).sum(0) != 0)
        tmp = data_numpy[:, index].copy()
        data_numpy *= 0
        data_numpy[:, :tmp.shape[1]] = tmp"""

        """data_numpy = data_numpy.transpose(3,1,2,0)  # M T V C
        xaxis = [7, 17]
        # print('parallel the bone in x plane between wrist(jpt 0) and TMCP(jpt 1) of the first person to the x axis')
        joint_left = data_numpy[0, 0, xaxis[0]].copy()
        joint_right = data_numpy[0, 0, xaxis[1]].copy()
        # axis = np.cross(joint_right - joint_left, [1, 0, 0])
        joint_left[2] = 0
        joint_right[2] = 0  # rotate by zaxis
        axis = np.cross(joint_right - joint_left, [1, 0, 0])
        angle = tools.angle_between(joint_right - joint_left, [1, 0, 0])
        matrix_x = tools.rotation_matrix(axis, angle)
        for i_p, person in enumerate(data_numpy):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    data_numpy[i_p, i_f, i_j] = np.dot(matrix_x, joint)
        data_numpy = data_numpy.transpose(3,1,2,0)  # C T V M"""

        # Xử lý Bone Stream: Bone Stream = hiệu giữa hai khớp xương
        if self.bone_stream:
            ori_data = data_numpy
            for v1, v2 in (
                (5, 6),
                (5, 7),
                (6, 8),
                (8, 10),
                (7, 9),
                (9, 11),
                (12, 13),
                (12, 14),
                (12, 16),
                (12, 18),
                (12, 20),
                (14, 15),
                (16, 17),
                (18, 19),
                (20, 21),
                (22, 23),
                (22, 24),
                (22, 26),
                (22, 28),
                (22, 30),
                (24, 25),
                (26, 27),
                (28, 29),
                (30, 31),
                (10, 12),
                (11, 22),
            ):
                data_numpy[:, :, v2 - 5, :] = (
                    ori_data[:, :, v2 - 5, :] - ori_data[:, :, v1 - 5, :]
                )

        # Xử lý Motion Stream: Motion Stream = hiệu giữa 2 frame liên tiếp
        if self.motion_stream:
            T = data_numpy.shape[1]
            ori_data = data_numpy
            for t in range(T - 1):
                data_numpy[:, t, :, :] = ori_data[:, t + 1, :, :] - ori_data[:, t, :, :]
            data_numpy[:, T - 1, :, :] = 0

        # if self.random_choose:
        #    data_numpy = tools.random_choose(data_numpy, self.window_size)

        if self.random_choose:
            data_numpy = tools.random_sample_np(data_numpy, self.window_size)
        else:
            data_numpy = tools.uniform_sample_np(data_numpy, self.window_size)

        """if self.random_choose:
            # data_numpy = uniform_sample_np(data_numpy, self.final_size)
            data_numpy = tools.random_choose_simple(data_numpy, self.window_size, center=True)
        else:
            data_numpy = tools.random_choose_simple(data_numpy, self.window_size)"""

        # nếu là random_mirror thì áp dụng kĩ thuật mirroring augmentation
        if self.random_mirror:
            if random.random() > self.random_mirror_p:
                assert data_numpy.shape[2] == 27
                data_numpy = data_numpy[:, :, flip_index, :]
                if self.is_vector:
                    data_numpy[0, :, :, :] = -data_numpy[0, :, :, :]
                else:
                    data_numpy[0, :, :, :] = (
                        512 - data_numpy[0, :, :, :]
                    )  # input size 512*512

        # nếu là chuẩn hóa dữ liệu thì chuẩn hóa dữ liệu bằng cách trừ giá trị trung bình và chia cho độ lệch chuẩn
        if self.normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            assert data_numpy.shape[0] == 3
            
            # nếu là vector thì chuẩn hóa theo chiều thứ 3 bằng cách trừ giá trị trung bình và chia cho độ lệch chuẩn
            if self.is_vector:
                data_numpy[0, :, 0, :] = data_numpy[0, :, 0, :] - data_numpy[
                    0, :, 0, 0
                ].mean(axis=0)
                data_numpy[1, :, 0, :] = data_numpy[1, :, 0, :] - data_numpy[
                    1, :, 0, 0
                ].mean(axis=0)
            # nếu không phải vector thì chuẩn hóa theo chiều thứ 2 bằng cách trừ giá trị trung bình và chia cho độ lệch chuẩn
            else:
                data_numpy[0, :, :, :] = data_numpy[0, :, :, :] - data_numpy[
                    0, :, 0, 0
                ].mean(axis=0)
                data_numpy[1, :, :, :] = data_numpy[1, :, :, :] - data_numpy[
                    1, :, 0, 0
                ].mean(axis=0)

        # nếu là random_shift thì áp dụng kĩ thuật random shift bằng cách thêm hoặc bớt các frame 0 vào đầu hoặc cuối chuỗi
        if self.random_shift:
            if not self.bone_stream:
                if self.is_vector:
                    data_numpy[0, :, 0, :] += random.random() * 20 - 10.0
                    data_numpy[1, :, 0, :] += random.random() * 20 - 10.0
                else:
                    data_numpy[0, :, :, :] += random.random() * 20 - 10.0
                    data_numpy[1, :, :, :] += random.random() * 20 - 10.0

        # if self.random_shift:
        #     data_numpy = tools.random_shift(data_numpy)

        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        if self.lap_pe:
            from torch_geometric.data import Data

            data = torch.tensor(data_numpy).float()
            data = rearrange(
                data, "c t v m -> (t v) (c m)", c=3, m=1, t=self.window_size, v=27
            )
            data = self.transform(
                Data(
                    x=data,
                    pos=data[:, :2],
                    edge_index=self.edge_index,
                    num_nodes=self.num_nodes,
                    y=torch.tensor([label], dtype=torch.int64),
                    EigVecs=self.eig_vecs,
                    EigVals=self.eig_vals,
                    window=self.window_size,
                )
            )
            return data, label, index

        # data_numpy[0,:] = data_numpy[0,:]/256
        # data_numpy[1,:] = data_numpy[1,:]/256
        return data_numpy, label, index

    """Hàm này kiểm tra xem nhãn thực tế có nằm trong Top-K dự đoán hay không"""
    def top_k(self, score, top_k):
        rank = score.argsort() # sắp xếp dự đoán theo thứ tự tăng dần
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)] # kiểm tra nhãn thực tế có nằm trong top-k dự đoán hay không
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    """Hàm này tính accuracy theo từng lớp"""
    def per_class_acc_top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [
            l in rank[i, -top_k:] for i, l in enumerate(self.label)
        ] # hit_top_k: lưu trữ kết quả đúng/sai cho từng mẫu.
        acc = [0 for c in range(self.num_class)]
        # tính accuracy cho từng lớp 
        for c in range(self.num_class):
            hit_label = [l == c for l in self.label] # hit_label: lưu trữ kết quả đúng/sai cho từng lớp.
            acc[c] = np.sum(
                np.array(hit_top_k).astype(np.float32)
                * np.array(hit_label).astype(np.float32)
            ) / self.label.count(c)
        return np.mean(acc)

"""Giúp import module bằng string thay vì phải gọi import module trực tiếp"""
def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    """
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    """
    import matplotlib.pyplot as plt

    # khởi tạo dataloader để load dữ liệu
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )

    # nếu vid không rỗng thì chỉ hiển thị một mẫu dữ liệu
    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split(".")[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape) # reshape data về dạng (1, C, T, V, M)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape # N: số lượng mẫu, C: số lượng kênh, T: số lượng frame, V: số lượng joints, M: số lượng người

        # khởi tạo figure để vẽ dữ liệu, ion là để hiển thị động tác theo thời gian
        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            
            # hiển thị dữ liệu 3D
            ax = fig.add_subplot(111, projection="3d")
        else: # hiển thị dữ liệu 2D
            ax = fig.add_subplot(111)

        # nếu không có đồ thị khớp xương, hiển thị từng khớp riêng lẻ bằng các điểm màu
        if graph is None:
            p_type = ["b.", "g.", "r.", "c.", "m.", "y.", "k.", "k.", "k.", "k."] # màu của từng khớp
            pose = [ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    # pose[m]: cập nhật tọa độ x, y của từng khớp theo thời gian T
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001) # tạo hiệu ứng chuyển động khi chạy qua từng frame
        else: # nếu có đồ thị skeleton, vẽ kết nối giữa các khớp
            p_type = ["b-", "g-", "r-", "c-", "m-", "y-", "k-", "k-", "k-", "k-"]
            import sys
            from os import path

            # thêm đường dẫn của thư mục cha vào sys.path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
            )
            # import class Graph từ đường dẫn graph để lấy đồ thị skeleton
            G = import_class(graph)()
            edge = G.inward # xác định các cặp khớp được kết nối (edge)
            pose = []
            # vẽ kết nối giữa các khớp
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    # nếu là 3d thì vẽ kết nối giữa các khớp theo 3 chiều (thêm trục z)
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                plt.savefig(str(t) + ".jpg")
                plt.pause(0.01)


if __name__ == "__main__":
    import os

    # sau này sẽ thay đổi đoạn này cho phù hợp với dữ liệu của mình
    os.environ["DISPLAY"] = "localhost:10.0"
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = "graph.ntu_rgb_d.Graph"
    test(data_path, label_path, vid="S004C001P003R001A032", graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
