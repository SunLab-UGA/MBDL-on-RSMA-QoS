from du_func import MyNetwork,Google_loss
from data import data_loading_OOD
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from comp_func import ssl_loss
import pandas as pd
import numpy
import os
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

def curve_plt(curve_list,str):
    plt.figure(figsize=(10, 5))
    plt.plot(curve_list, label=str)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
dataset_label = "org_lp"
MyDataset = data_loading_OOD("./optdata/")
test_ASR_record_iteration = []
violation_rate_record_iteration = []
# sample_scale = 50

for iteration_exp in range(1):
    if iteration_exp == 0:
        train_size = 600
        num_epochs = 30

    test_size = len(MyDataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(MyDataset, [train_size, test_size])
    train_dataset=DataLoader(dataset=train_dataset, batch_size=100,shuffle=True)
    test_dataset=DataLoader(dataset=test_dataset, batch_size=10000, shuffle=True)
    layer_num=5
    model = MyNetwork(layer_num)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    model.train()
    lossfunction = Google_loss()
    print("Training Start...")
    loss_history=[]
    last_layer_histroy=[]
    train_hat_WSR_recording=[]
    train_expected_WSR_recording=[]
    test_hat_WSR_recording=[]
    test_expected_WSR_recording=[]
    train_WSR_MAPE_recording=[]
    test_WSR_MAPE_recording=[]
    torch.autograd.set_detect_anomaly(True)
    loss_part1_data = []
    loss_part2_data = []
    loss_part1_last_data = []
    loss_part2_last_data = []
    for epoch in range(num_epochs):
        print(epoch)
        batch = 0
        for R1c, R2c, R3c, rc1, rc2, rc3, H1, H2, H3, f1, f2, f3, r1, r2, r3, sigma, P_max,v0, v1, v2, v3 in train_dataset:
            batch =  batch+1
            R1c = torch.unsqueeze(R1c, 1)
            R2c = torch.unsqueeze(R2c, 1)
            R3c = torch.unsqueeze(R3c, 1)

            rc1 = torch.unsqueeze(rc1, 1)
            rc2 = torch.unsqueeze(rc2, 1)
            rc3 = torch.unsqueeze(rc3, 1)

            H1 = torch.unsqueeze(H1, 2)
            H2 = torch.unsqueeze(H2, 2)
            H3 = torch.unsqueeze(H3, 2)

            v0 = torch.unsqueeze(v0, 2)
            v1 = torch.unsqueeze(v1, 2)
            v2 = torch.unsqueeze(v2, 2)
            v3 = torch.unsqueeze(v3, 2)

            P_max = torch.unsqueeze(P_max, 1)
            sigma = torch.unsqueeze(sigma, 1)

            f1 = torch.unsqueeze(f1, 1)
            f2 = torch.unsqueeze(f2, 1)
            f3 = torch.unsqueeze(f3, 1)
            sum_f = f1 + f2 + f3
            r1 = torch.unsqueeze(r1, 1)
            r2 = torch.unsqueeze(r2, 1)
            r3 = torch.unsqueeze(r3, 1)

            unique_vec = torch.stack((f1/sum_f, f2/sum_f, f3/sum_f, sigma ), dim=1)
            unique_vec = torch.squeeze(unique_vec, 3)
            unique_vec = unique_vec.type(torch.float)

            optimizer.zero_grad()
            layer_outputs,counter_num,count_hist_gamma,count_hist_g = model(H1,H2,H3,f1,f2,f3,r1,r2,r3,rc1,rc2,rc3,sigma,P_max,unique_vec,iteration_exp)
            loss,last_layer,WSR_MAPE_last_layer,loss_part1,loss_part2,loss_part1_last,loss_part2_last,ASR_layer,counter_num_layer = lossfunction.forward(R1c,R2c,R3c,v0,v1,v2,v3,layer_outputs,H1,H2,H3,f1,f2,f3,r1,r2,r3,sigma,layer_num,P_max)
            last_layer_histroy.append(last_layer)
            loss_history.append(loss.detach().numpy())
            train_hat_WSR_recording.append(loss.detach().numpy())
            train_WSR_MAPE_recording.append(WSR_MAPE_last_layer)
            loss_part1_data.append(loss_part1)
            loss_part2_data.append(loss_part2)
            loss_part1_last_data.append(loss_part1_last)
            loss_part2_last_data.append(loss_part2_last)
            loss.backward()
            optimizer.step()
    model_type = 'du'
    torch.save(model.state_dict(), './model_save/'+model_type+'_model.pth')
    print("training terminated, testing started...........................\n")
    model.eval()
    test_loss_history=[]
    test_loss_history_lastlayer=[]
    layer_asr = []
    layer_vio = []
    counter_num_history = []
    ASR_layer_all = []
    counter_num_layer_all = []
    test_size = len(MyDataset)
    train_dataset, test_dataset = torch.utils.data.random_split(MyDataset, [0, test_size])
    test_dataset=DataLoader(dataset=test_dataset, batch_size=10000, shuffle=True)
    lossfunction = ssl_loss()
    with torch.no_grad():
        print(epoch)
        batch = 0
        for R1c, R2c, R3c, rc1, rc2, rc3, H1, H2, H3, f1, f2, f3, r1, r2, r3, sigma, P_max,v0, v1, v2, v3 in test_dataset:
            batch =  batch+1
            R1c = torch.unsqueeze(R1c, 1)
            R2c = torch.unsqueeze(R2c, 1)
            R3c = torch.unsqueeze(R3c, 1)

            rc1 = torch.unsqueeze(rc1, 1)
            rc2 = torch.unsqueeze(rc2, 1)
            rc3 = torch.unsqueeze(rc3, 1)

            H1 = torch.unsqueeze(H1, 2)
            H2 = torch.unsqueeze(H2, 2)
            H3 = torch.unsqueeze(H3, 2)

            v0 = torch.unsqueeze(v0, 2)
            v1 = torch.unsqueeze(v1, 2)
            v2 = torch.unsqueeze(v2, 2)
            v3 = torch.unsqueeze(v3, 2)

            P_max = torch.unsqueeze(P_max, 1)
            sigma = torch.unsqueeze(sigma, 1)

            f1 = torch.unsqueeze(f1, 1)
            f2 = torch.unsqueeze(f2, 1)
            f3 = torch.unsqueeze(f3, 1)
            sum_f = f1 + f2 + f3
            r1 = torch.unsqueeze(r1, 1)
            r2 = torch.unsqueeze(r2, 1)
            r3 = torch.unsqueeze(r3, 1)

            unique_vec = torch.stack((f1/sum_f, f2/sum_f, f3/sum_f, sigma ), dim=1)
            unique_vec = torch.squeeze(unique_vec, 3)
            unique_vec = unique_vec.type(torch.float)

            optimizer.zero_grad()
            layer_outputs,counter_num,count_hist_gamma,count_hist_g = model(1,H1,H2,H3,f1,f2,f3,r1,r2,r3,rc1,rc2,rc3,sigma,P_max,unique_vec,iteration_exp)
            [R1c_pre,R2c_pre,R3c_pre,v1_pre, v2_pre, v3_pre,negative_count,counter_num,v0_pre]= layer_outputs[-1]
            
            loss, WSR_MAPE_last_layer, violation, sample_vio= lossfunction.forward(R1c,R2c,R3c,v0,v1,v2,v3,v0_pre, v1_pre, v2_pre,v3_pre,R1c_pre,R2c_pre,R3c_pre,H1,H2,H3,f1,f2,f3,r1,r2,r3,rc1,rc2,rc3,sigma,P_max)
            
            print(WSR_MAPE_last_layer)
            print("equations violation: ",violation)
            print("sample violation: ",sample_vio)
    
    # train_loss_recording_df = pd.DataFrame(numpy.array(train_hat_WSR_recording)).T
    # train_loss_recording_df.to_excel('./result_excel/'+dataset_label+'layer'+str(layer_num)+'train_loss.xlsx', index=False, header=False)
    
    # train_WSR_MAPE_recording_df = pd.DataFrame(numpy.array(train_WSR_MAPE_recording)).T
    # train_WSR_MAPE_recording_df.to_excel('./result_excel/'+dataset_label+'layer'+str(layer_num)+'train_MAPE.xlsx', index=False, header=False)
    
    # part1_df = pd.DataFrame(numpy.array(loss_part1_data)).T
    # part1_df.to_excel('./result_excel/'+'part1_data.xlsx', index=False, header=False)
    
    # part2_df = pd.DataFrame(numpy.array(loss_part2_data)).T
    # part2_df.to_excel('./result_excel/'+'part2_data.xlsx', index=False, header=False)
    
    # part1_last_df = pd.DataFrame(numpy.array(loss_part1_last_data)).T
    # part1_last_df.to_excel('./result_excel/'+'part1_last.xlsx', index=False, header=False)
    
    # part2_last_df = pd.DataFrame(numpy.array(loss_part2_last_data)).T
    # part2_last_df.to_excel('./result_excel/'+'part2_last.xlsx', index=False, header=False)
    