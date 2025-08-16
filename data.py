import numpy
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)
class MyDataset(Dataset):

    def __init__(self,R1c,R2c,R3c,rc1,rc2,rc3,H1,H2,H3,f1,f2,f3,r1,r2,r3,sigma,P_max,v0,v1,v2,v3):
        self.R1c = R1c
        self.R2c = R2c
        self.R3c = R3c

        self.rc1 = rc1
        self.rc2 = rc2
        self.rc3 = rc3

        self.H1 = H1
        self.H2 = H2
        self.H3 = H3

        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

        self.r1 = r1
        self.r2 = r2
        self.r3 = r3

        self.sigma = sigma
        self.P_max = P_max

        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def __len__(self):
        return len(self.f1)
    def __getitem__(self, idx):
        R1c = self.R1c[idx]
        R2c = self.R2c[idx]
        R3c = self.R3c[idx]

        rc1 = self.rc1[idx]
        rc2 = self.rc2[idx]
        rc3 = self.rc3[idx] 

        H1 = self.H1[idx]
        H2 = self.H2[idx]
        H3 = self.H3[idx]

        f1 = self.f1[idx]
        f2 = self.f2[idx]
        f3 = self.f3[idx]

        r1 = self.r1[idx]
        r2 = self.r2[idx]
        r3 = self.r3[idx]

        sigma = self.sigma[idx]
        P_max = self.P_max[idx]

        v0 = self.v0[idx]
        v1 = self.v1[idx]
        v2 = self.v2[idx]
        v3 = self.v3[idx]
        R1c = torch.Tensor(R1c)
        R2c = torch.Tensor(R2c)
        R3c = torch.Tensor(R3c)

        rc1 = torch.Tensor(rc1)
        rc2 = torch.Tensor(rc2)
        rc3 = torch.Tensor(rc3)

        H1 = torch.Tensor(H1)
        H2 = torch.Tensor(H2)
        H3 = torch.Tensor(H3)

        f1 = torch.Tensor(f1)
        f2 = torch.Tensor(f2)
        f3 = torch.Tensor(f3)

        r1 = torch.Tensor(r1)
        r2 = torch.Tensor(r2)
        r3 = torch.Tensor(r3)

        v0 = torch.Tensor(v0)
        v1 = torch.Tensor(v1)
        v2 = torch.Tensor(v2)
        v3 = torch.Tensor(v3)

        P_max = torch.Tensor(P_max)
        sigma = torch.Tensor(sigma)

        return  R1c,R2c,R3c,rc1,rc2,rc3,H1,H2,H3,f1,f2,f3,r1,r2,r3,sigma,P_max,v0,v1,v2,v3

def data_loading_OOD(str):
    R1c = pd.read_excel(str+"R1c.xlsx", header=None)
    R2c = pd.read_excel(str+"R2c.xlsx", header=None)
    R3c = pd.read_excel(str+"R3c.xlsx", header=None)

    rc1 = pd.read_excel(str+"rc1.xlsx", header=None)
    rc2 = pd.read_excel(str+"rc2.xlsx", header=None)
    rc3 = pd.read_excel(str+"rc3.xlsx", header=None)

    H1_re = pd.read_excel(str+"H1_re.xlsx", header=None)
    H2_re = pd.read_excel(str+"H2_re.xlsx", header=None)
    H3_re = pd.read_excel(str+"H3_re.xlsx", header=None)
    H1_im = pd.read_excel(str+"H1_im.xlsx", header=None)
    H2_im = pd.read_excel(str+"H2_im.xlsx", header=None)
    H3_im = pd.read_excel(str+"H3_im.xlsx", header=None)

    f1 = pd.read_excel(str+"f1.xlsx", header=None)
    f2 = pd.read_excel(str+"f2.xlsx", header=None)
    f3 = pd.read_excel(str+"f3.xlsx", header=None)

    r1 = pd.read_excel(str+"r1.xlsx", header=None)
    r2 = pd.read_excel(str+"r2.xlsx", header=None)
    r3 = pd.read_excel(str+"r3.xlsx", header=None)

    sigma = pd.read_excel(str+"sigma.xlsx", header=None)
    P_max = pd.read_excel(str+"P_max.xlsx", header=None)

    v0_re = pd.read_excel(str+"v0_re.xlsx", header=None)
    v1_re = pd.read_excel(str+"v1_re.xlsx", header=None)
    v2_re = pd.read_excel(str+"v2_re.xlsx", header=None)
    v3_re = pd.read_excel(str+"v3_re.xlsx", header=None)

    v0_im = pd.read_excel(str+"v0_im.xlsx", header=None)
    v1_im = pd.read_excel(str+"v1_im.xlsx", header=None)
    v2_im = pd.read_excel(str+"v2_im.xlsx", header=None)
    v3_im = pd.read_excel(str+"v3_im.xlsx", header=None)

    R1c_array = R1c.to_numpy()
    R2c_array = R2c.to_numpy()
    R3c_array = R3c.to_numpy()

    rc1_array = rc1.to_numpy()
    rc2_array = rc2.to_numpy()
    rc3_array = rc3.to_numpy()

    H1_re_array = H1_re.to_numpy()
    H2_re_array = H2_re.to_numpy()
    H3_re_array = H3_re.to_numpy()
    H1_im_array = H1_im.to_numpy()
    H2_im_array = H2_im.to_numpy()
    H3_im_array = H3_im.to_numpy()

    f1_array = f1.to_numpy()
    f2_array = f2.to_numpy()
    f3_array = f3.to_numpy()

    r1_array = r1.to_numpy()
    r2_array = r2.to_numpy()
    r3_array = r3.to_numpy()

    sigma_array = sigma.to_numpy()
    P_max_array = P_max.to_numpy()

    v0_re_array = v0_re.to_numpy()
    v1_re_array = v1_re.to_numpy()
    v2_re_array = v2_re.to_numpy()
    v3_re_array = v3_re.to_numpy()

    v0_im_array = v0_im.to_numpy()
    v1_im_array = v1_im.to_numpy()
    v2_im_array = v2_im.to_numpy()
    v3_im_array = v3_im.to_numpy()

    H1 = H1_re_array + 1j * H1_im_array
    H2 = H2_re_array + 1j * H2_im_array
    H3 = H3_re_array + 1j * H3_im_array

    v0 = v0_re_array + 1j * v0_im_array
    v1 = v1_re_array + 1j * v1_im_array
    v2 = v2_re_array + 1j * v2_im_array
    v3 = v3_re_array + 1j * v3_im_array

    R1c = torch.tensor(R1c_array)
    R2c = torch.tensor(R2c_array)
    R3c = torch.tensor(R3c_array)

    rc1 = torch.tensor(rc1_array)
    rc2 = torch.tensor(rc2_array)
    rc3 = torch.tensor(rc3_array)

    H1 = torch.tensor(H1)
    H2 = torch.tensor(H2)
    H3 = torch.tensor(H3)

    v0 = torch.tensor(v0)
    v1 = torch.tensor(v1)
    v2 = torch.tensor(v2)
    v3 = torch.tensor(v3)

    H1 = H1.type(torch.complex128)
    H2 = H2.type(torch.complex128)
    H3 = H3.type(torch.complex128)

    v0 = v0.type(torch.complex128)
    v1 = v1.type(torch.complex128)
    v2 = v2.type(torch.complex128)
    v3 = v3.type(torch.complex128)

    f1 = torch.tensor(f1_array)
    f2 = torch.tensor(f2_array)
    f3 = torch.tensor(f3_array)

    r1 = torch.tensor(r1_array)
    r2 = torch.tensor(r2_array)
    r3 = torch.tensor(r3_array)


    sigma = torch.tensor(sigma_array)
    P_max = torch.tensor(P_max_array)

    data_struct=MyDataset(R1c,R2c,R3c,rc1,rc2,rc3,H1,H2,H3,f1,f2,f3,r1,r2,r3,sigma,P_max,v0,v1,v2,v3)
    return data_struct
