import numpy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)
def print_grad(grad):
    print(grad)

def loss_iterative(layer_info,f1,f2,f3,r1,r2,r3,H1,H2,H3,H1H,H2H,H3H,sigma,layer_seq,expected_WSR,P_max):
    [R1c_hat,R2c_hat,R3c_hat,v1_hat, v2_hat, v3_hat,negative_count,counter_num,v0_hat]=layer_info
    v0H_hat = torch.transpose(v0_hat, 1, 2).conj()
    v1H_hat = torch.transpose(v1_hat, 1, 2).conj()
    v2H_hat = torch.transpose(v2_hat, 1, 2).conj()
    v3H_hat = torch.transpose(v3_hat, 1, 2).conj()

    hat_WSR =   f1 * (R1c_hat+torch.log2(1+torch.real(
        H1H @ v1_hat @ v1H_hat @ H1)/ torch.real(
                sigma + H1H @ v2_hat @ v2H_hat @ H1 + H1H @ v3_hat @ v3H_hat @ H1))) \
              + f2 * (R2c_hat+torch.log2(1+torch.real(
        H2H @ v2_hat @ v2H_hat @ H2)/ torch.real(
                sigma + H2H @ v1_hat @ v1H_hat @ H2 + H2H @ v3_hat @ v3H_hat @ H2))) \
              + f3 * (R3c_hat+torch.log2(1+torch.real(
        H3H @ v3_hat @ v3H_hat @ H3)/ torch.real(
                sigma + H3H @ v2_hat @ v2H_hat @ H3 + H3H @ v1_hat @ v1H_hat @ H3)))
    loss = -torch.sum(hat_WSR) * torch.log2(
        torch.tensor(layer_seq)) / len(H1)+(negative_count)* torch.log2(
        torch.tensor(layer_seq)) / len(H1)*100
    loss_part1 = -torch.sum(hat_WSR) * torch.log2(
        torch.tensor(layer_seq)) / len(H1)
    loss_part2 = (negative_count)* torch.log2(
        torch.tensor(layer_seq)) / len(H1)*100
    WSR_MAPE = torch.sum(hat_WSR/expected_WSR)/len(hat_WSR)
    last_layer_dis_EU = torch.sum(hat_WSR / expected_WSR) / len(hat_WSR)
    return loss,WSR_MAPE,last_layer_dis_EU,loss_part1.detach().numpy(),loss_part2.detach().numpy(),counter_num
class Google_loss(nn.Module):
    def __init__(self):
        super(Google_loss, self).__init__()

    def forward(self,R1c,R2c,R3c,v0,v1,v2,v3,layer_outputs,H1,H2,H3,f1,f2,f3,r1,r2,r3,sigma,layer_num,P_max):
        v0H = torch.transpose(v0, 1, 2).conj()
        v1H = torch.transpose(v1, 1, 2).conj()
        v2H = torch.transpose(v2, 1, 2).conj()
        v3H = torch.transpose(v3, 1, 2).conj()
        H1H = torch.transpose(H1, 1, 2).conj()
        H2H = torch.transpose(H2, 1, 2).conj()
        H3H = torch.transpose(H3, 1, 2).conj()
        expected_WSR =  f1 * (R1c+torch.log2(1+torch.real(H1H @ v1 @ v1H @ H1)/ torch.real(
                sigma + H1H @ v2 @ v2H @ H1 + H1H @ v3 @ v3H @ H1))) \
                      + f2 * (R2c+torch.log2(1+torch.real(H2H @ v2 @ v2H @ H2)/ torch.real(
                sigma + H2H @ v1 @ v1H @ H2 + H2H @ v3 @ v3H @ H2))) \
                      + f3 * (R3c+torch.log2(1+torch.real(H3H @ v3 @ v3H @ H3)/ torch.real(
                sigma + H3H @ v2 @ v2H @ H3 + H3H @ v1 @ v1H @ H3)))
        loss=0
        loss_part1 = 0
        loss_part2 = 0
        ASR_layer = []
        counter_num_layer = []
        for i in range(len(layer_outputs)):
            term_loss,WSR_MAPE,last_layer_dis_EU,loss_part1_item,loss_part2_item,counter_num=loss_iterative(layer_outputs[i], f1, f2, f3,r1,r2,r3, H1, H2, H3, H1H, H2H, H3H, sigma, i + 2, expected_WSR,
                               P_max)
            loss=term_loss+loss
            loss_part1 = loss_part1+loss_part1_item
            loss_part2 = loss_part2 + loss_part2_item
            last_layer_ad, WSR_MAPE_last_layer, last_layer_dis_EU, loss_part1_last, loss_part2_last,counter_num = loss_iterative(
                layer_outputs[i], f1, f2, f3, r1, r2, r3, H1, H2, H3, H1H, H2H, H3H, sigma, 2, expected_WSR, P_max)
            counter_num_layer.append(counter_num)
            ASR_layer.append(last_layer_dis_EU.detach().numpy())
        loss=loss/len(layer_outputs)
        last_layer_ad,WSR_MAPE_last_layer,last_layer_dis_EU,loss_part1_last,loss_part2_last,counter_num=loss_iterative(layer_outputs[i],f1,f2,f3,r1,r2,r3,H1,H2,H3,H1H,H2H,H3H,sigma,2,expected_WSR,P_max)
        last_layer = last_layer_dis_EU.detach().numpy()

        WSR_MAPE_last_layer = WSR_MAPE_last_layer.detach().numpy()
        print("\n last layer output")
        print(last_layer)

        print("weighted layer output")
        print(loss.detach().numpy())
        return loss,last_layer,WSR_MAPE_last_layer,loss_part1,loss_part2,loss_part1_last,loss_part2_last,ASR_layer,counter_num_layer

class RSMA_layer(nn.Module):
    def __init__(self):
        super(RSMA_layer, self).__init__()
        self.weights_v0 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v1 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v2 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v3 = nn.Parameter(torch.randn(1, 1, 1) / 20)

        self.weights_v01 = nn.Parameter(torch.rand(1, 1, 1) / 20)
        self.weights_v02 = nn.Parameter(torch.rand(1, 1, 1) / 20)
        self.weights_v03 = nn.Parameter(torch.rand(1, 1, 1) / 20)
        self.weights_v03 = nn.Parameter(torch.rand(1, 1, 1) / 20)

        self.weights_v11 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v12 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v13 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v14 = nn.Parameter(torch.randn(1, 1, 1) / 20)

        self.weights_v21 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v22 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v23 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v24 = nn.Parameter(torch.randn(1, 1, 1) / 20)

        self.weights_v31 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v32 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v33 = nn.Parameter(torch.randn(1, 1, 1) / 20)
        self.weights_v34 = nn.Parameter(torch.randn(1, 1, 1) / 20)

    def forward(self,H1, H2, H3, f1, f2, f3, r1, r2, r3, rc1, rc2, rc3, sigma, P_max,v0, v1, v2, v3, unique_vec,H1H,H2H,H3H,iteration_exp):

        v0H = torch.transpose(v0, 1, 2).conj()
        v1H = torch.transpose(v1, 1, 2).conj()
        v2H = torch.transpose(v2, 1, 2).conj()
        v3H = torch.transpose(v3, 1, 2).conj()

        H1V1 = H1H@v1@v1H@H1
        H1V2 = H1H@v2@v2H@H1
        H1V3 = H1H@v3@v3H@H1

        H2V2 = H2H@v2@v2H@H2
        H2V1 = H2H@v1@v1H@H2
        H2V3 = H2H@v3@v3H@H2 

        H3V3 = H3H@v3@v3H@H3 
        H3V1 = H3H@v1@v1H@H3
        H3V2 = H3H@v2@v2H@H3
        z01 = ((H1H@v0))/torch.real(sigma+H1V1+H1V2+H1V3)
        z02 = ((H2H@v0))/torch.real(sigma+H2V1+H2V2+H2V3)
        z03 = ((H3H@v0))/torch.real(sigma+H3V1+H3V2+H3V3)
        z1 = ((H1H@v1))/torch.real(sigma+H1V2+H1V3)
        z2 = ((H2H@v2))/torch.real(sigma+H2V1+H2V3)
        z3 = ((H3H@v3))/torch.real(sigma+H3V1+H3V2)
        z01_conj = torch.conj(z01)
        z02_conj = torch.conj(z02)
        z03_conj = torch.conj(z03)

        z1_conj = torch.conj(z1)
        z2_conj = torch.conj(z2)
        z3_conj = torch.conj(z3)
        GD_v0 = self.weights_v01 *torch.real(
            2 * z01_conj  * H1 )/(1 + 2* torch.real(z01_conj * H1H @ v0) - z01 * z01_conj*(sigma + H1V1 + H1V2 + H1V3))+\
            self.weights_v02 *torch.real(
            2 * z02_conj  * H2 )/(1 + 2* torch.real(z02_conj * H2H @ v0) - z02 * z02_conj*(sigma + H2V1 + H2V2 + H2V3))+\
            self.weights_v03 *torch.real(
            2 * z03_conj  * H3 )/(1 + 2* torch.real(z03_conj * H3H @ v0) - z03 * z03_conj*(sigma + H3V1 + H3V2 + H3V3))
        GD_v1 = 2 * (self.weights_v11 * f1 * (H1 * z1_conj) / (1 + torch.real(
            torch.real(2 * z1_conj * H1H @ v1) - z1_conj * (sigma + H1V2 + H1V3) * z1))) \
                - 2 * self.weights_v12 * f2 * H2 @ H2H * z2 * z2_conj / (1 + torch.real(
            torch.real(2 * z2_conj * H2H @ v2) - z2_conj * (sigma + H2V1 + H2V3) * z2)) @ v1 \
                - 2 * self.weights_v13 * f3 * H3 @ H3H * z3 * z3_conj / (1 + torch.real(
            torch.real(2 * z3_conj * H3H @ v3) - z3_conj * (sigma + H3V2 + H3V1) * z3)) @ v1 
        
        assert not torch.isnan(GD_v1).any(), "Tensor contains NaNs"

        GD_v2 = 2 * (self.weights_v21 * f2 * (H2 * z2_conj) / (1 + torch.real(
            torch.real(2 * z2_conj * H2H @ v2) - z2_conj * (sigma + H2V1 + H2V3) * z2))) \
                - 2 * self.weights_v22 * f1  * H1 @ H1H * z1 * z1_conj / ((1 + torch.real(
            torch.real(2 * z1_conj * H1H @ v1) - z1_conj * (sigma + H1V2 + H1V3) * z1))) @ v2 \
                - 2 * self.weights_v23 * f3  * H3 @ H3H * z3 * z3_conj / ((1 + torch.real(
            torch.real(2 * z3_conj * H3H @ v3) - z3_conj * (sigma + H3V2 + H3V1) * z3))) @ v2 
        
        assert not torch.isnan(GD_v2).any(), "Tensor contains NaNs"
        GD_v3 = 2 * (self.weights_v31 * f3 * (H3 * z3_conj) / (1 + torch.real(
            torch.real(2 * z3_conj * H3H @ v3) - z3_conj * (sigma + H3V2 + H3V1) * z3))) \
                - 2 * self.weights_v32 * f2 * H2 @ H2H * z2 * z2_conj / (1 + torch.real(
            torch.real(2 * z2_conj * H2H @ v2) - z2_conj * (sigma + H2V1 + H2V3) * z2)) @ v3 \
                - 2 * self.weights_v33 * f1 * H1 @ H1H * z1 * z1_conj / (1 + torch.real(
            torch.real(2 * z1_conj * H1H @ v1) - z1_conj * (sigma + H1V2 + H1V3) * z1)) @ v3 
        assert not torch.isnan(GD_v3).any(), "Tensor contains NaNs"

        v0_update = v0 + (self.weights_v0)*GD_v0
        assert not torch.isnan(v0_update).any(), "Tensor contains NaNs"
        v1_update = v1 + (self.weights_v1)*GD_v1
        assert not torch.isnan(v1_update).any(), "Tensor contains NaNs"
        v2_update = v2 + (self.weights_v2)*GD_v2
        assert not torch.isnan(v2_update).any(), "Tensor contains NaNs"
        v3_update = v3 + (self.weights_v3)*GD_v3
        assert not torch.isnan(v3_update).any(), "Tensor contains NaNs"
        v0H_update = torch.transpose(v0_update, 1, 2).conj()
        v1H_update = torch.transpose(v1_update, 1, 2).conj()
        v2H_update = torch.transpose(v2_update, 1, 2).conj()
        v3H_update = torch.transpose(v3_update, 1, 2).conj()
        v0_norm = torch.norm(v0_update, dim=1, keepdim=True)
        v1_norm = torch.norm(v1_update, dim=1, keepdim=True)
        v2_norm = torch.norm(v2_update, dim=1, keepdim=True)
        v3_norm = torch.norm(v3_update, dim=1, keepdim=True)
        v0_power = torch.square(v0_norm)
        v1_power = torch.square(v1_norm)
        v2_power = torch.square(v2_norm)
        v3_power = torch.square(v3_norm)
        a = torch.cat((v1_power, v2_power, v3_power,v0_power), dim=1)
        r = torch.cat((r1, r2, r3), dim=1)
        rc_sum = rc1+rc2+rc3
        rc_exp = torch.exp2(rc1+rc2+rc3)-1
        rc = torch.cat((rc_exp, rc_exp, rc_exp), dim=1)
        A = torch.zeros((len(H1),7,4),dtype=torch.float64)
        v0_n = v0_update / v0_norm
        v1_n = v1_update / v1_norm
        v2_n = v2_update / v2_norm
        v3_n = v3_update / v3_norm

        v0H_n = v0H_update / v0_norm
        v1H_n = v1H_update / v1_norm
        v2H_n = v2H_update / v2_norm
        v3H_n = v3H_update / v3_norm
        A[:, 0, 0] = (H1H @ v1_n @ v1H_n  @ H1).squeeze(-1).squeeze(-1)
        A[:, 0, 1] = (-r1 * H1H @ v2_n @ v2H_n @ H1).squeeze(-1).squeeze(-1)
        A[:, 0, 2] = (-r1 * H1H @ v3_n @ v3H_n @ H1).squeeze(-1).squeeze(-1)

        A[:, 1, 0] = (-r2 * H2H @ v1_n @ v1H_n @ H2).squeeze(-1).squeeze(-1)
        A[:, 1, 1] = (H2H @ v2_n @ v2H_n @ H2).squeeze(-1).squeeze(-1)
        A[:, 1, 2] = (-r2 * H2H @ v3_n @ v3H_n @ H2).squeeze(-1).squeeze(-1)

        A[:, 2, 0] = (-r3 * H3H @ v1_n @ v1H_n @ H3).squeeze(-1).squeeze(-1)
        A[:, 2, 1] = (-r3 * H3H @ v2_n @ v2H_n @ H3).squeeze(-1).squeeze(-1)
        A[:, 2, 2] = (H3H @ v3_n @ v3H_n @ H3).squeeze(-1).squeeze(-1)

        A[:, 3, 0] = (-rc_exp * H1H @ v1_n @ v1H_n @ H1).squeeze(-1).squeeze(-1)
        A[:, 3, 1] = (-rc_exp * H1H @ v2_n @ v2H_n @ H1).squeeze(-1).squeeze(-1)
        A[:, 3, 2] = (-rc_exp * H1H @ v3_n @ v3H_n @ H1).squeeze(-1).squeeze(-1)
        A[:, 3, 3] = (H1H @ v0_n @ v0H_n @ H1).squeeze(-1).squeeze(-1)

        A[:, 4, 0] = (-rc_exp * H2H @ v1_n @ v1H_n @ H2).squeeze(-1).squeeze(-1)
        A[:, 4, 1] = (-rc_exp * H2H @ v2_n @ v2H_n @ H2).squeeze(-1).squeeze(-1)
        A[:, 4, 2] = (-rc_exp * H2H @ v3_n @ v3H_n @ H2).squeeze(-1).squeeze(-1)
        A[:, 4, 3] = (H2H @ v0_n @ v0H_n @ H2).squeeze(-1).squeeze(-1)

        A[:, 5, 0] = (-rc_exp * H3H @ v1_n @ v1H_n @ H3).squeeze(-1).squeeze(-1)
        A[:, 5, 1] = (-rc_exp * H3H @ v2_n @ v2H_n @ H3).squeeze(-1).squeeze(-1)
        A[:, 5, 2] = (-rc_exp * H3H @ v3_n @ v3H_n @ H3).squeeze(-1).squeeze(-1)
        A[:, 5, 3] = (H3H @ v0_n @ v0H_n @ H3).squeeze(-1).squeeze(-1)

        A[:, 6, 0] = -(10 * v1_norm / v1_norm).squeeze(-1).squeeze(-1)
        A[:, 6, 1] = -(10 * v2_norm / v2_norm).squeeze(-1).squeeze(-1)
        A[:, 6, 2] = -(10 * v3_norm / v3_norm).squeeze(-1).squeeze(-1)
        A[:, 6, 3] = -(10 * v0_norm / v0_norm).squeeze(-1).squeeze(-1)
        A = torch.real(A)
        onehotcons = torch.zeros(len(A), 7, 7)
        onehotcons[:, 0, 0] = 1
        onehotcons[:, 1, 1] = 1
        onehotcons[:, 2, 2] = 1
        onehotcons[:, 3, 3] = 1
        onehotcons[:, 4, 4] = 1
        onehotcons[:, 5, 5] = 1
        onehotcons[:, 6, 6] = 1
        
        A = torch.cat((A, onehotcons), dim=2)
        n1 = r*sigma.repeat_interleave(3, dim=1)
        n2 = rc*sigma.repeat_interleave(3, dim=1)
        n = torch.cat((n1,n2),dim=1)
        n = torch.cat((n,-P_max*10),dim=1)
        zero_pad = torch.zeros(len(v1_update), 7,1)
        a = torch.cat((a,zero_pad),dim=1)
        AH = torch.transpose(A, dim1=-1, dim0=-2)
        AAH_pinv = torch.linalg.pinv(A@AH)
        lambda_vec = torch.relu(AAH_pinv @ (A @ a - n))
        W_projection =(a - AH @lambda_vec)
        w1,w2,w3,w0,w_1,w_2,w_3,w_0,w_1c,w_2c,w_3c = torch.split(W_projection, split_size_or_sections=1, dim=1)
        v0_new = v0_update/v0_norm * (torch.sqrt(torch.relu(w0)+2e-6))
        v1_new = v1_update/v1_norm * (torch.sqrt(torch.relu(w1)+2e-6))
        v2_new = v2_update/v2_norm * (torch.sqrt(torch.relu(w2)+2e-6))
        v3_new = v3_update/v3_norm * (torch.sqrt(torch.relu(w3)+2e-6))

        v0H_new = torch.transpose(v0_new, 1, 2).conj()
        v1H_new = torch.transpose(v1_new, 1, 2).conj()
        v2H_new = torch.transpose(v2_new, 1, 2).conj()
        v3H_new = torch.transpose(v3_new, 1, 2).conj()

        c1 = torch.log2(1 + (torch.real(
            H1H @ v0_new @ v0H_new @ H1 / (sigma + H1H @ v1_new @ v1H_new @ H1 + H1H @ v2_new @ v2H_new @ H1 + H1H @ v3_new @ v3H_new @ H1))))
        c2 = torch.log2(1 + (torch.real(
            H2H @ v0_new @ v0H_new @ H2 / (sigma + H2H @ v1_new @ v1H_new @ H2 + H2H @ v2_new @ v2H_new @ H2 + H2H @ v3_new @ v3H_new @ H2))))
        c3 = torch.log2(1 + (torch.real(
            H3H @ v0_new @ v0H_new @ H3 / (sigma + H3H @ v1_new @ v1H_new @ H3 + H3H @ v2_new @ v2H_new @ H3 + H3H @ v3_new @ v3H_new @ H3))))
        bound_c = torch.stack((c1, c2, c3), dim=2)

        global_min_c, min_indices_c = torch.min(bound_c, dim=2)
        upperbound = torch.where(min_indices_c==0,c1,c3)
        upperbound = torch.where(min_indices_c==1,c2,upperbound)

        
        max_values = torch.stack((f1, f2, f3), dim=2)
        global_max, max_indices = torch.max(max_values, dim=2)
        rc_stack0 = torch.stack((upperbound*rc1/rc_sum, upperbound*rc2/rc_sum, upperbound*rc3/rc_sum), dim=1).squeeze(-1)
        mask1 = max_indices==0
        mask2 = max_indices==1
        mask3 = max_indices==2


        rc1_vio = ( c1- rc1-rc2-rc3)
        # print(torch.sum(rc1_vio +1e-5< 0 ))

        rc2_vio = ( c2 - rc1-rc2-rc3)
        # print(torch.sum(rc2_vio +1e-5< 0 ))

        rc3_vio = ( c3 - rc1-rc2-rc3)
        # print(torch.sum(rc3_vio +1e-5< 0 ))

        viomsk1 = rc1_vio > 0 
        viomsk2 = rc2_vio > 0 
        viomsk3 = rc3_vio > 0 
        rc_stack1 = torch.where(mask1&viomsk1, torch.stack((upperbound - rc2- rc3, rc2, rc3), dim=1).squeeze(-1),rc_stack0)
        rc_stack2 = torch.where(mask2&viomsk2, torch.stack((rc1, upperbound - rc1- rc3, rc3), dim=1).squeeze(-1),rc_stack1)
        rc_stack3 = torch.where(mask3&viomsk3, torch.stack((rc1, rc2, upperbound - rc1- rc2), dim=1).squeeze(-1),rc_stack2)
        R1c = rc_stack3[:,0].unsqueeze(-1)
        R2c = rc_stack3[:,1].unsqueeze(-1)
        R3c = rc_stack3[:,2].unsqueeze(-1)
        
        assert not torch.isnan(rc_stack3).any(), "Tensor contains NaNs"
        if torch.any(w0)<0:
            print("0")
        if torch.any(w1)<0:
            print("1")
        if torch.any(w2)<0:
            print("2")
        if torch.any(w3)<0:
            print("3")
        if torch.isinf(v1_new).any():
            print("b[leavaing] inf")
        if torch.isinf(v2_new).any():
            print("b[i] inf")
        if torch.isinf(v3_new).any():
            print("factor inf")
        if torch.isnan(v1_new).any():
            print("b[leavaing] nan")
        if torch.isnan(v2_new).any():
            print("b[i] nan")
        if torch.isnan(v3_new).any():
            print("factor nan")
        if torch.any(v1_new == 0):
            raise ValueError("0 tensor found.")
        if torch.any(v2_new == 0):
            raise ValueError("0 tensor found.")
        if torch.any(v3_new == 0):
            raise ValueError("0 tensor found.")
       
        count_hist_gamma = []
        count_hist_g = []
        power_vio = torch.real(
            v0H_new @ v0_new + v1H_new @ v1_new + v2H_new @ v2_new  + v3H_new @ v3_new - P_max)
        r1_vio = ( torch.real(
            H1H @ v1_new @ v1H_new @ H1) / torch.real(
            sigma + H1H @ v2_new @ v2H_new @ H1 + H1H @ v3_new @ v3H_new @ H1) - r1)
        r2_vio = ( torch.real(
            H2H @ v2_new @ v2H_new @ H2) / torch.real(
            sigma + H2H @ v1_new @ v1H_new @ H2 + H2H @ v3_new @ v3H_new @ H2) - r2)
        r3_vio = ( torch.real(
            H3H @ v3_new @ v3H_new @ H3) / torch.real(
            sigma + H3H @ v2_new @ v2H_new @ H3 + H3H @ v1_new @ v1H_new @ H3) - r3)
        negative_count = torch.sum(torch.relu(-r1_vio)) + torch.sum(torch.relu(-r2_vio)) + torch.sum(torch.relu(-r3_vio))+ torch.sum(torch.relu(-rc1_vio))+ torch.sum(torch.relu(-rc2_vio)) + torch.sum(torch.relu(-rc3_vio)) + torch.sum(torch.relu(rc1-R1c))+ torch.sum(torch.relu(rc2-R2c))+ torch.sum(torch.relu(rc3-R3c)) + torch.sum(torch.relu(power_vio))
        counter_num = (torch.sum(r3_vio < 0 )+torch.sum(r2_vio < 0 )+torch.sum(r1_vio < 0 )+torch.sum(rc1_vio < 0 )+torch.sum(rc2_vio < 0 )+torch.sum(rc3_vio < 0 )+torch.sum(R1c < rc1 )+torch.sum(R2c < rc2 )+torch.sum(R3c < rc3 )+torch.sum(power_vio>0))/len(H2)
        return R1c,R2c,R3c,v0_new, v1_new,v2_new,v3_new,counter_num,negative_count,count_hist_gamma,count_hist_g

class MyNetwork(nn.Module):
    def __init__(self,num_layers):
        super(MyNetwork, self).__init__()
        self.layers = nn.ModuleList([RSMA_layer() for _ in range(num_layers)])
    def forward(self,H1,H2,H3,f1,f2,f3,r1,r2,r3,rc1,rc2,rc3,sigma,P_max,unique_vec,iteration_exp):
        layer_outputs = []
        count=0
        batch_size=len(H1)
        v0 = (H1+H2+H3)/ 10
        v1 = H1/ 10
        v2 = H2/ 10
        v3 = H3/ 10
        H1H = torch.transpose(H1, 1, 2).conj()
        H2H = torch.transpose(H2, 1, 2).conj()
        H3H = torch.transpose(H3, 1, 2).conj()
        for layer in self.layers:
            count = count + 1
            R1c,R2c,R3c,v0,v1, v2, v3,counter_num,negative_count,count_hist_gamma,count_hist_g = layer(H1, H2, H3, f1, f2, f3, r1, r2, r3, rc1, rc2, rc3, sigma, P_max,v0, v1, v2, v3, unique_vec,H1H,H2H,H3H,iteration_exp)
            layer_outputs.append((R1c,R2c,R3c,v1, v2, v3,negative_count,counter_num,v0))
        return layer_outputs,counter_num,count_hist_gamma,count_hist_g
