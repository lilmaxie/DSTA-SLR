import sys

sys.path.extend(["../"]) # nếu muốn import từ các thư mục khác, thêm đường dẫn vào đây
from graph import tools # dùng hàm get_spatial_graph từ tools.py để tạo ma trận kề cho mô hình GCN

"""
Mục tiêu: 
- Xây dựng cấu trúc đồ thị (graph) cho bộ xương skeleton có 27 khớp.
- Định nghĩa cách các khớp kết nối với nhau trong không gian.
- Tạo ma trận kề (Adjacency Matrix) để sử dụng trong mô hình GCN.
- Hỗ trợ nhiều định dạng dữ liệu khác nhau, gồm WLASL, Kinetics, NTU.

Quan trọng vì:
- Đồ thị skeleton giúp mô hình học được quan hệ giữa các khớp.
- NẾU DỮ LIỆU BÊN ĐỘNG TÁC DƯỠNG SINH CÓ SỐ KHỚP KHÁC NHAU, THÌ CẦN SỬA FILE NÀY.
"""

class Graph:
    """ labeling_mode: chọn cách tạo ma trận kề, có thể là "spatial" hoặc "temporal"
        graph: chọn loại dữ liệu, có thể là "wlasl", "kinetics" hoặc "ntu"
        NẾU MUỐN DÙNG ĐỒ THỊ TÙY CHỈNH CHO DƯỠNG SINH, CÓ THỂ THÊM LỰA CHỌN MỚI Ở ĐÂY
    """
    def __init__(self, labeling_mode="spatial", graph="wlasl"):

        if graph == "wlasl":
            num_node = 27 # 27 khớp quan trọng nhất (nếu dataset mình khác thì thay đổi số num_node)
            self_link = [(i, i) for i in range(num_node)] # mỗi khớp kết nối với chính nó (liên kết tự thân)
            # Các cặp khớp kết nối với nhau theo chiều từ trong ra ngoài, dựa trên hệ tọa độ COCO format
            # NẾU CẦN ĐIỀU CHỈNH ĐỒ THỊ CHO DƯỠNG SINH, ĐÂY CŨNG LÀ PHẦN CẦN THAY ĐỔI
            inward_ori_index = [
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
            ]
            
            # tạo danh sách inward, outward và neighbor từ inward_ori_index
            inward = [(i - 5, j - 5) for (i, j) in inward_ori_index] # hướng vào (từ gốc đến ngọn)
            outward = [(j, i) for (i, j) in inward] # hướng ra (ngược inward)
            neighbor = inward + outward # tất cả kết nối giữa các khớp
        
        # CẦN PHẢI TẠO THÊM MỘT KHỔI MỚI CHO DỮ LIỆU DƯỠNG SINH 
        
        elif graph == "kinetics":
            num_node = 18 # theo đúng định dạng của kinetics
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [
                (4, 3),
                (3, 2),
                (7, 6),
                (6, 5),
                (13, 12),
                (12, 11),
                (10, 9),
                (9, 8),
                (11, 5),
                (8, 2),
                (5, 1),
                (2, 1),
                (0, 1),
                (15, 0),
                (14, 0),
                (17, 15),
                (16, 14),
            ]
            inward = inward_ori_index
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
        elif graph == "ntu":
            num_node = 25 # theo đúng định dạng của NTU
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [
                (1, 2),
                (2, 21),
                (3, 21),
                (4, 3),
                (5, 21),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 21),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 1),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 1),
                (18, 17),
                (19, 18),
                (20, 19),
                (22, 23),
                (23, 8),
                (24, 25),
                (25, 12),
            ]
            inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward

        """
        Tạo ma trận kề adjacency matrix từ các thông số trên
        - Lưu thông tin số khớp và danh sách kết nối.
        - Tạo ma trận kề bằng get_adjacency_matrix().
        - NẾU THAY ĐỔI DANH SÁCH KHỚP HOẶC LIÊN KẾT, MA TRÂN KỀ SẼ THAY ĐỔI THEO.
        """
        self.num_node = num_node # số khớp
        self.self_link = self_link # liên kết tự thân
        self.inward = inward # hướng vào
        self.outward = outward # hướng ra
        self.neighbor = neighbor # kết nối giữa các khớp

        self.A = self.get_adjacency_matrix(labeling_mode)

    # hàm tạo ma trận kề, tạo bằng cách gọi hàm get_spatial_graph từ tools.py
    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        # nếu là spatial thì tạo ma trận kề theo hướng không gian
        if labeling_mode == "spatial":
            A = tools.get_spatial_graph( # muốn chỉnh hàm này thì vô tools.py sửa
                self.num_node, self.self_link, self.inward, self.outward
            )
        else:
            raise ValueError()
        return A


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph("spatial").get_adjacency_matrix()
    # hiển thị ma trận kề dưới dạng ảnh xám
    # dùng để kiểm tra xem đồ thị có đúng không
    for i in A:
        plt.imshow(i, cmap="gray")
        plt.show()
    print(A)
