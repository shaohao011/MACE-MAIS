import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 256
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

    def forward(self, x):

        A = self.attention(x)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        Z = torch.mm(A, x)  # ATTENTION_BRANCHESxM
        
        return Z


class GatedAttention(nn.Module):
    def __init__(self,in_dim=256,embed_dim=128):
        super(GatedAttention, self).__init__()
        self.M = in_dim
        self.L = embed_dim
        self.ATTENTION_BRANCHES = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)


    def forward(self, x):
        # x : [B, N, M]
        # print(x.shape)
        A_V = self.attention_V(x)  # B N L
        A_U = self.attention_U(x)  # B N L
        A = self.attention_w(A_V * A_U) # B N 1
        A = torch.transpose(A, 2, 1)  # B 1 N
        A = F.softmax(A, dim=2)  # get score of each token
        # print(A.shape,x.shape)
        Z = torch.bmm(A, x)  # B M so we merge the token
        return Z,A
    
if __name__=="__main__":
    import torch
    data = torch.randn((3,4,1024)).cuda()
    gate = GatedAttention(in_dim=1024,embed_dim=512).cuda()
    outputs,A  = gate(data)
    print(outputs.shape,A[0])
    
