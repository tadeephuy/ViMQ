import torch.nn as nn
import torch


class BiaffineLayer(nn.Module):
    def __init__(self, inSize1, inSize2, classSize, dropout=0.3):
        super(BiaffineLayer, self).__init__()

        self.bilinearMap = nn.Parameter(torch.FloatTensor(inSize1 + 1, classSize, inSize2 + 1))
        self.classSize = classSize

    def forward(self, x1, x2):
        # [b, n, v1] -> [b*n, v1]
        # print("BIAFFINEPARA:", self.bilinearMap)
        batch_size = x1.shape[0]
        bucket_size = x1.shape[1]

        x1 = torch.cat((x1,torch.ones([batch_size, bucket_size, 1]).to(x1.device)), axis=2)
        x2 = torch.cat((x2, torch.ones([batch_size, bucket_size, 1]).to(x2.device)), axis=2)
        # Static shape info
        vector_set_1_size = x1.shape[-1]
        vector_set_2_size = x2.shape[-1]

        # [b, n, v1] -> [b*n, v1]
        vector_set_1 = x1.reshape((-1, vector_set_1_size))

        # [v1, r, v2] -> [v1, r*v2]
        bilinear_map = self.bilinearMap.reshape((vector_set_1_size, -1))

        # [b*n, v1] x [v1, r*v2] -> [b*n, r*v2]
        bilinear_mapping = torch.matmul(vector_set_1, bilinear_map)

        # [b*n, r*v2] -> [b, n*r, v2]
        bilinear_mapping = bilinear_mapping.reshape(
            (batch_size, bucket_size * self.classSize, vector_set_2_size))

        # [b, n*r, v2] x [b, n, v2]T -> [b, n*r, n]
        bilinear_mapping = torch.matmul(bilinear_mapping, x2.transpose(1, -1))

        # [b, n*r, n] -> [b, n, r, n]
        bilinear_mapping = bilinear_mapping.reshape(
            (batch_size, bucket_size, self.classSize, bucket_size))

        return bilinear_mapping.transpose(-2, -1)