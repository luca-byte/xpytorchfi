import random
import struct
import logging
import json
import sys, os
import pandas as pd
import math
from collections import defaultdict

import matplotlib.pyplot as plt

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.insert(1, 'pytorchfi')
import torch
import pytorchfi
import numpy as np
import pytorchfi.core as core
from scipy.spatial.distance import euclidean
from pytorchfi.util import random_value, relative_iou, setup_dicts, compute_iou, compute_mAP
from pytorchfi.core import *
from pytorchfi.neuron_error_models import *
from torchmetrics.detection.mean_ap import MeanAveragePrecision


logger=logging.getLogger("Fault_injection") 
logger.setLevel(logging.DEBUG) 

# _bf_inj_w_mask=0
# _layer=0


# def float_to_hex(f):
#     h=hex(struct.unpack('<I', struct.pack('<f', f))[0])
#     return h[2:len(h)]


# def hex_to_float(h):
#     return float(struct.unpack(">f",struct.pack(">I",int(h,16)))[0])

# def int_to_float(h):
#     return float(struct.unpack(">f",struct.pack(">I",h))[0])
    


# def _log_faults(Log_string):
#     with open("./FSIM_logs/Fsim_log.log",'a+') as logfile:
#         logfile.write(Log_string+'\n')

# def bit_flip_weight_inj(pfi: core.FaultInjection, layer, k, c_in, kH, kW, inj_mask):
#     global _bf_inj_w_mask
#     global _layer
#     _bf_inj_w_mask=inj_mask
#     _layer=layer
#     return pfi.declare_weight_fault_injection(
#         function=_bit_flip_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW
#     )

# def _bit_flip_weight(data, location):
#     orig_data=data[location].item()
#     data_32bit=int(float_to_hex(data[location].item()),16)

#     #print(_bf_inj_w_mask)

#     corrupt_32bit=data_32bit ^ int(_bf_inj_w_mask[0])
#     corrupt_val=int_to_float(corrupt_32bit)
#     #print(data_32bit,_bf_inj_w_mask,corrupt_32bit, orig_data, corrupt_val)
#     log_msg=f"F_descriptor: Layer:{_layer}, (K, C, H, W):{location}, BitMask:{_bf_inj_w_mask[0]}, Ffree_Weight:{data_32bit}, Faulty_weight:{corrupt_32bit}"
#     logger.info(log_msg)
#     _log_faults(log_msg)
#     return corrupt_val

#     # k, c, H, W

        
        
    # def update_fault_dictionary(self,key,val):           
    #     if self._layer not in self.fault_dictionary:
    #         self.fault_dictionary[self._layer]={}            
    #     if self._kK not in self.fault_dictionary[self._layer]:
    #         self.fault_dictionary[self._layer][self._kK]={}
    #     if self._kC not in  self.fault_dictionary[self._layer][self._kK]:
    #         self.fault_dictionary[self._layer][self._kK][self._kC]={}
    #     if self._kH not in  self.fault_dictionary[self._layer][self._kK][self._kC]:
    #         self.fault_dictionary[self._layer][self._kK][self._kC][self._kH]={}
    #     if self._kW not in  self.fault_dictionary[self._layer][self._kK][self._kC][self._kH]:                            
    #         self.fault_dictionary[self._layer][self._kK][self._kC][self._kH][self._kW]={}  
    #     if self._inj_mask not in  self.fault_dictionary[self._layer][self._kK][self._kC][self._kH][self._kW]:
    #         self.fault_dictionary[self._layer][self._kK][self._kC][self._kH][self._kW][self._inj_mask]={}
    #     self.fault_dictionary[self._layer][self._kK][self._kC][self._kH][self._kW][self._inj_mask][key]=val

        # k, c, H, W
def float_to_bin(f):
    h=hex(struct.unpack('<I', struct.pack('<f', f))[0])
    weight_bin_clean="{:032b}".format(int(h[2:len(h)],16))
    return(weight_bin_clean)

def float_flip(f,pos,mode="01"):
    h=hex(struct.unpack('<I', struct.pack('<f', f))[0])
    d=int(h[2:len(h)],16)
    if(mode=="01"):
        d=d|(2**pos)
    elif(mode=="10"):
        d=d&((2**pos)^(int('FFFFFFFF',16)))
    else:
        d=d
    return (float(struct.unpack(">f",struct.pack(">I",d))[0]))

def weight_distribution(pfi_model:FaultInjection, **kwargs):    
    if kwargs:
        layer_num=kwargs.get('layer')
        path=kwargs.get('path')
        
        
        if(os.path.exists(os.path.join(path,"weights_distribution_DNN.csv"))):
            D_df = pd.read_csv(os.path.join(path,"weights_distribution_DNN.csv"))
            p=D_df['p'].values.tolist()
            return(p)
        
        
        layer_idx=0

        weight_distribution_dict={}
        weight_distribution_dict[0]=[0]*32
        weight_distribution_dict[1]=[0]*32

        weights=[]

        # pfi_model.print_pytorchfi_layer_summary()

        for layer in pfi_model.original_model.modules():
            if isinstance(layer, tuple(pfi_model._inj_layer_types)):
                if(layer_idx==layer_num):
                    weight_dimention=pfi_model.get_weights_dim(layer_idx)
                    weight_shape=list(pfi_model.get_weights_size(layer_idx))
                    print(layer_idx,weight_shape)
                    for dim1 in range(weight_shape[0]):
                        for dim2 in range(weight_shape[1]):
                            if(weight_dimention>2):
                                for dim3 in range(weight_shape[2]):
                                    for dim4 in range(weight_shape[3]):
                                        weight_idx = tuple(
                                            [
                                                dim1,
                                                dim2,
                                                dim3,
                                                dim4,
                                            ]
                                        )
                                        weight_value = layer.weight[weight_idx].item()
                                        weights.append(weight_value)                                        
                                        weight_bin_clean=float_to_bin(weight_value)
                                        for bitidx in range(0,32):
                                            if(int(weight_bin_clean[bitidx]) not in weight_distribution_dict):
                                                weight_distribution_dict[int(weight_bin_clean[bitidx])][bitidx]=1
                                            else:
                                                weight_distribution_dict[int(weight_bin_clean[bitidx])][bitidx]+=1
                                        
                            else:
                                weight_idx = tuple(
                                            [
                                                dim1,
                                                dim2
                                            ]
                                        )
                                weight_value = layer.weight[weight_idx].item()
                                weights.append(weight_value)                                        
                                weight_bin_clean=float_to_bin(weight_value)
                                for bitidx in range(0,32):
                                    if(int(weight_bin_clean[bitidx]) not in weight_distribution_dict):
                                        weight_distribution_dict[int(weight_bin_clean[bitidx])][bitidx]=1
                                    else:
                                        weight_distribution_dict[int(weight_bin_clean[bitidx])][bitidx]+=1                                                 
                layer_idx+=1   
    zeros=[0]*32
    ones=[0]*32
    for i in range(32):
        zeros[i]=(weight_distribution_dict[0][31-i])
        ones[i]=(weight_distribution_dict[1][31-i])
    
    weigth_hist_df=pd.DataFrame({"weights":weights})

    D01avg=[0]*32
    D01max=[0]*32
    D01min=[0]*32
    D10avg=[0]*32
    D10max=[0]*32
    D10min=[0]*32

    Davg_bit=[0]*32

    for bit in range(32):
        # weigth_hist_df[f"valid_bits0_{bit}"]=(weigth_hist_df["weights"].apply(lambda x: int(float_to_bin(x),2)&2**bit))
        # df_bit_zero=weigth_hist_df.loc[(weigth_hist_df[f"valid_bits0_{bit}"]==0)]
        weigth_hist_df[f"bit{bit}_0_1"]=weigth_hist_df["weights"].apply(float_flip,pos=bit,mode="01") 
        tmpdf=pd.DataFrame()
        tmpdf["diff"]=(weigth_hist_df[f"bit{bit}_0_1"]-weigth_hist_df["weights"])
        print(tmpdf["diff"].mean())
        D01max[bit]=tmpdf["diff"].max() if not pd.isna(tmpdf["diff"].max()) else 0
        D01avg[bit]=tmpdf["diff"].mean() if not pd.isna(tmpdf["diff"].mean()) else 0
        D01min[bit]=tmpdf["diff"].min() if not pd.isna(tmpdf["diff"].min()) else 0     

        # weigth_hist_df[f"valid_bits1_{bit}"]=(weigth_hist_df["weights"].apply(lambda x: int(float_to_bin(x),2)&2**bit))
        # df_bit_one=weigth_hist_df.loc[(weigth_hist_df[f"valid_bits1_{bit}"]!=0)]
        weigth_hist_df[f"bit{bit}_1_0"]=weigth_hist_df["weights"].apply(float_flip,pos=bit,mode="10") 
        tmpdf=pd.DataFrame()
        tmpdf["diff"]=(weigth_hist_df[f"bit{bit}_1_0"]-weigth_hist_df["weights"])

        D10max[bit]=tmpdf["diff"].max() if not pd.isna(tmpdf["diff"].max()) else 0
        D10avg[bit]=tmpdf["diff"].mean() if not pd.isna(tmpdf["diff"].mean()) else 0
        D10min[bit]=tmpdf["diff"].min() if not pd.isna(tmpdf["diff"].min()) else 0

        Davg_bit[bit]=D01avg[bit]*zeros[bit]+D10avg[bit]*ones[bit]

    #weigth_hist_df.to_csv(f"{path}/D_avg.csv")
    Davg_bit[30]=0

    D_avg_max=max(Davg_bit)
    D_avg_min=min(Davg_bit)

    p=[0]*32
    a=0
    b=0.5
    for bit in range(32):
        p[bit]=a+(Davg_bit[bit]-D_avg_min)*(b-a)/(D_avg_max-D_avg_min)

    p[30]=0.5
    fig,ax = plt.subplots()
    fig.set_size_inches(12,5)
    ax=plt.bar(x=[x for x in range(32)],height=p)
    plt.yscale('log')
    fig.savefig(f"{path}/p_{layer_num}.jpg")


    index=[f"bit{i}" for i in range(32)]
    df=pd.DataFrame({"zeros":zeros,"ones":ones,"bits":index})
    D_df=pd.DataFrame({"D01max":D01max,"D01avg":D01avg,"D01min":D01min,
                       "D10max":D10max,"D10avg":D10avg,"D10min":D10min,
                       "zeros":zeros,"ones":ones,"bits":index,
                       "p": p, "Davg_bit":Davg_bit, "bits":index})
    
    D_df.to_csv(os.path.join(path,"weights_distribution_DNN.csv"))
    
    print(f"max_weight_val = {max(weights)}")
    print(f"min_weight_val = {min(weights)}")

    fig,ax=plt.subplots()
    #ax=df.plot.bar(x="bits",rot=0)
    ax=weigth_hist_df.hist(ax=ax, column="weights",bins=50)
    plt.yscale('log')
    fig.savefig(f"{path}/weights_distribution_layer_{layer_num}.jpg")

    maxval=(weigth_hist_df["weights"].abs()).max()
    minval=(weigth_hist_df["weights"].abs()).min()

    #plt.rcParams["figure.figsize"] = [12,5]
    fig,ax=plt.subplots()
    fig.set_size_inches(12,5)
    fig.text(0.2,0.95,f"max_weight_val = {maxval}; bin = {float_to_bin(maxval)}")
    fig.text(0.2,0.9, f"min_weight_val = {minval}; bin = {float_to_bin(minval)}")
    ax=df.plot.bar(x="bits",rot=0,ax=ax)
    #ax.set_yscale('log')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig.savefig(f"{path}/weights_bit_distribution_layer_{layer_num}.jpg")    
    return(p)

def generate_fault_list_sbfm(path,pfi_model:FaultInjection, **kwargs):
    T=1.64485362695147  #confidence level
    E=0.05 # 0.01             #error margin
    P=0.5               # Perrror
    MSB_inection=31
    LSB_injection=20
    fault_list=[]   
    f_list=pd.DataFrame()             
    if kwargs:                               
        fault_list_file=kwargs.get('f_list_file')
        if not os.path.exists(os.path.join(path,fault_list_file)):                  
            f_list=pd.DataFrame(columns=['layer','kernel','channel','row','col','bitmask'])            
            layer_param=kwargs.get('layer')
            kK_param=kwargs.get('kernel')
            kC_param=kwargs.get('channel')
            # pfi_model.print_pytorchfi_layer_summary()            
            print(pfi_model.get_total_layers())
            if layer_param!= None:
                layr=layer_param
            else:
                layr=random.randint(0,pfi_model.get_total_layers()-1)
            weight_shape=list(pfi_model.get_weights_size(layr))

            # print(len(weight_shape))
            if kK_param != None:
                N=1
            else:
                N=weight_shape[0]
            
            if kC_param != None:
                N*=1
            else:
                N*=weight_shape[1]
            if(len(weight_shape)==4):
                N=N*weight_shape[2]*weight_shape[3]*(MSB_inection-LSB_injection+1)
            else:
                N=N*(MSB_inection-LSB_injection+1)

            # print(N)
            n=int(N/(1+(E**2)*(N-1)/((T**2)*P*(1-P))))
            # n = 5
            print(f'**********************: {n}')
            # n = 2
            #print(n)                
            i=0
            while i<n:
                if kK_param != None:
                    k=kK_param
                else:
                    k=random.randint(0,weight_shape[0]-1)                
                if kC_param != None:
                    c=kC_param
                else:
                    c=random.randint(0,weight_shape[1]-1)
                if(len(weight_shape)==4):
                    h=random.randint(0,weight_shape[2]-1)
                    w=random.randint(0,weight_shape[3]-1)
                else:
                    h=None
                    w=None

                mask=2**(random.randint(LSB_injection,MSB_inection))                
                fault=[layr,k,c,h,w,mask]
                if fault not in fault_list :
                    fault_list.append(fault)
                    fault_dict={'layer':layr,'kernel':k,'channel':c,'row':h,'col':w,'bitmask':mask}
                    new_row=pd.DataFrame(fault_dict, index=[0])
                    f_list=pd.concat([f_list, new_row],ignore_index=True, sort=False)                                                        
                    i+=1
            f_list.to_csv(os.path.join(path,fault_list_file),sep=',')
        else:
            f_list = pd.read_csv(os.path.join(path,fault_list_file),index_col=[0]) 
    return(f_list)


def generate_fault_list_sbfm_fails(path,pfi_model:FaultInjection, **kwargs):
    T=2.576  #confidence level
    E=0.01              #error margin
    #P=0.5               # Perrror
    MSB_inection=31
    LSB_injection=19
    fault_list=[]   
    f_list=pd.DataFrame()             
    if kwargs:                               
        fault_list_file=kwargs.get('f_list_file')
        if not os.path.exists(os.path.join(path,fault_list_file)):                  
            f_list=pd.DataFrame(columns=['layer','kernel','channel','row','col','bitmask'])            
            layer_param=kwargs.get('layer')
            kK_param=kwargs.get('kernel')
            kC_param=kwargs.get('channel')
            # pfi_model.print_pytorchfi_layer_summary()            
            print(pfi_model.get_total_layers())
            if layer_param!= None:
                layr=layer_param
            else:
                layr=random.randint(0,pfi_model.get_total_layers()-1)
            weight_shape=list(pfi_model.get_weights_size(layr))
            
            # P = kwargs.get('p')
            # (P) = weight_distribution(pfi_model=pfi_model,layer=layr,path=path) 
            
            P=[0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.000001,
               0.00001,
               0.0005,
               0.0005,
               0.0005,
               0.005,
               0.005,
               0.05,
               0.05,
               0.05,
               0.01,
               0.01,
               0.01,
               0.5,
               0.05]

            N=1
            for num_dim in range(pfi_model.get_weights_dim(layr)):
                N*=weight_shape[num_dim]
            print(N)

            for bitdx in range(32):
                n=int(N/(1+(E**2)*(N-1)/((T**2)*P[bitdx]*(1-P[bitdx]))))
                print(n,N)
                i=0
                while i<n:
                    k=random.randint(0,weight_shape[0]-1)                
                    c=random.randint(0,weight_shape[1]-1)
                    h=random.randint(0,weight_shape[2]-1)
                    w=random.randint(0,weight_shape[3]-1)
                    mask=2**(bitdx)                
                    fault=[layr,k,c,h,w,mask]
                    if fault not in fault_list :
                        fault_list.append(fault)
                        fault_dict={'layer':layr,'kernel':k,'channel':c,'row':h,'col':w,'bitmask':mask}
                        new_row=pd.DataFrame(fault_dict, index=[0])
                        f_list=pd.concat([f_list, new_row],ignore_index=True, sort=False)                                                        
                        i+=1
            f_list.to_csv(os.path.join(path,fault_list_file),sep=',')
        else:
            f_list = pd.read_csv(os.path.join(path,fault_list_file),index_col=[0]) 
    return(f_list)


def generate_fault_neurons_tailing(path,pfi_model:FaultInjection, **kwargs):
    fault_list=[]   
    fault_dict={}
    f_list=pd.DataFrame()             
    if kwargs:                               
        fault_list_file=kwargs.get('f_list_file')
        if not os.path.exists(os.path.join(path,fault_list_file)):                  
            # f_list=pd.DataFrame(columns=['layer','kernel','channel','height','width','bitmask'])     
            Num_trials=kwargs.get('trials') 
            tail_bloc_y=kwargs.get('size_tail_y') 
            tail_bloc_x=kwargs.get('size_tail_x') 

            layer=kwargs.get('layers')
            # pfi_model.print_pytorchfi_layer_summary()            
            print(pfi_model.get_total_layers())

            # fault_dict['ber']=kwargs.get('BER')
            fault_dict['layer_start']=layer[0]
            if len(layer)>1:
                fault_dict['layer_stop']=layer[1]
            else:
                fault_dict['layer_stop']=layer[0]

           
            #fault_dict['neuron_fault_rate']=kwargs.get('neuron_fault_rate')
            
            fault_dict['size_tail_y']=tail_bloc_y
            fault_dict['size_tail_x']=tail_bloc_x

            b_rate_delta=kwargs.get('block_fault_rate_delta')
            b_rate_steps=kwargs.get('block_fault_rate_steps')

            n_rate_delta=kwargs.get('neuron_fault_rate_delta')
            n_rate_steps=kwargs.get('neuron_fault_rate_steps')
            
            if(b_rate_steps==None or b_rate_delta==None):
                b_rate_steps = 1
                b_rate_delta=0.01

            if(n_rate_steps==None or n_rate_delta==None):
                n_rate_steps = 1
                n_rate_delta=0.01

            for bfr in range(1,b_rate_steps+1):
                fault_dict['block_fault_rate']=bfr*b_rate_delta
                for nfr in range(1,n_rate_steps+1):
                    n_rate=nfr*n_rate_delta
                    fault_dict['neuron_fault_rate']=n_rate
                    for bit_pos_fault in range(19,32):
                        for _ in (range(Num_trials)):  
                            fault_dict['bit_faulty_pos']=bit_pos_fault                                                                      
                            new_row=pd.DataFrame(fault_dict, index=[0])
                            f_list=pd.concat([f_list, new_row],ignore_index=True, sort=False)     

            f_list.to_csv(os.path.join(path,fault_list_file),sep=',')
        else:
            f_list = pd.read_csv(os.path.join(path,fault_list_file),index_col=[0]) 
    return(f_list)

def generate_fault_list_ber(path,pfi_model:FaultInjection, **kwargs):
    fault_list=[]   
    fault_dict={}
    f_list=pd.DataFrame()             
    if kwargs:                               
        fault_list_file=kwargs.get('f_list_file')
        if not os.path.exists(os.path.join(path,fault_list_file)):                  
            # f_list=pd.DataFrame(columns=['layer','kernel','channel','height','width','bitmask'])     
            BER=kwargs.get('BER')   
            Num_trials=kwargs.get('trials')     
            # pfi_model.print_pytorchfi_layer_summary()            
            print(pfi_model.get_total_layers())

            # fault_dict['ber']=kwargs.get('BER')

            if kwargs.get('layer')!= None:
                fault_dict['layer']=kwargs.get('layer')
            if kwargs.get('kernel') != None:
                fault_dict['kernel']=kwargs.get('kernel')
            if kwargs.get('channel')!=None:
                fault_dict['channel']=kwargs.get('channel')
            if kwargs.get('row')!=None:
                fault_dict['row']=kwargs.get('row')
            if kwargs.get('col')!=None:
                fault_dict['col']=kwargs.get('col')

            for i in (range(Num_trials)):
                for brate in range(0,BER+1):
                    fault_dict['ber']=brate
                    new_row=pd.DataFrame(fault_dict, index=[0])
                    f_list=pd.concat([f_list, new_row],ignore_index=True, sort=False)                                                        
                    i+=1
                f_list.to_csv(os.path.join(path,fault_list_file),sep=',')
        else:
            f_list = pd.read_csv(os.path.join(path,fault_list_file),index_col=[0]) 
    return(f_list.values.tolist())


def generate_error_list_neurons(pfi_model:FaultInjection,layer=-1,channel=-1,row=-1,col=-1):
    if layer == -1:
        layer = random.randint(0, pfi_model.get_total_layers() - 1)

    dim = pfi_model.get_layer_dim(layer)
    shape = pfi_model.get_layer_shape(layer)

    dim1_shape = shape[1]
    if channel==-1:
        dim1_rand = random.randint(0, dim1_shape - 1)
    else:
        dim1_rand=channel

    if dim > 2:
        dim2_shape = shape[2]
        if(row==-1):
            dim2_rand = random.randint(0, dim2_shape - 1)
        else:
            dim2_rand=row
    else:
        dim2_rand = None

    if dim > 3:
        dim3_shape = shape[3]
        if(col==-1):
            dim3_rand = random.randint(0, dim3_shape - 1)
        else:
            dim3_rand = col
    else:
        dim3_rand = None

    return (layer, dim1_rand, dim2_rand, dim3_rand)


def loc_neuron(layer=-1,dim=1,shape=[],BlockID_y=1,BlockID_x=1,Neuron_x=-1,Neuron_y=-1,tail_bloc_y=1,tail_bloc_x=1,):
    if Neuron_y==-1:
        dy=random.randint(0,tail_bloc_y-1)
    else:
        dy=Neuron_y
    if Neuron_x==-1:
        dx=random.randint(0,tail_bloc_x-1)
    else:
        dx=Neuron_x

    if dim>3:
        dim1_rand=int((BlockID_y*tail_bloc_y+dy)/tail_bloc_y)
        dim2_rand=int((BlockID_x*tail_bloc_x+dx)/shape[2])
        dim3_rand=int((BlockID_x*tail_bloc_x+dx)%shape[2])

        if(dim1_rand>=shape[1]):
            dim1_rand=shape[1]-1

        if(dim2_rand>=shape[2]):
            dim2_rand=shape[2]-1

        if(dim3_rand>=shape[3]):
            dim3_rand=shape[3]-1

    else:        
        dim1_rand=int((BlockID_y*tail_bloc_y+dy)/tail_bloc_y)
        dim2_rand=None
        dim3_rand=None

    return(layer, dim1_rand, dim2_rand, dim3_rand)


def generate_error_list_neurons_tails(pfi_model:FaultInjection,layer_i=-1,layer_n=-1,block_error_rate=1,neuron_fault_rate=0.001,tail_bloc_y=32,tail_bloc_x=32):
    if layer_i == -1:
        layer_i = random.randint(0, pfi_model.get_total_layers() - 1)
    if layer_n == -1:
       layer_n = layer_i

    locations=[]
    batch_order=[]
    fault_info={}
    for layer in range(layer_i,layer_n+1):
        dim = pfi_model.get_layer_dim(layer)
        shape = pfi_model.get_layer_shape(layer)

        GEMM_y=shape[1] 
        if(dim==2):
            GEMM_x=1    
        else:
            GEMM_x=shape[2]*shape[3]

        if GEMM_x*GEMM_y<tail_bloc_y*tail_bloc_x:
            tail_bloc_x=GEMM_x
            tail_bloc_y=GEMM_y
        else:
            if GEMM_y<tail_bloc_y and GEMM_x>=tail_bloc_x:
                tail_bloc_x=tail_bloc_y*tail_bloc_x/GEMM_y
                tail_bloc_y=GEMM_y
            elif GEMM_x<tail_bloc_x and GEMM_y>=tail_bloc_y:
                tail_bloc_y=tail_bloc_y*tail_bloc_x/GEMM_x
                tail_bloc_x=GEMM_x

        if(dim==2):
            if(GEMM_y==tail_bloc_y*tail_bloc_x):
                max_tail_y=0
            else:
                max_tail_y= int(GEMM_y/tail_bloc_y*tail_bloc_x)
            if(GEMM_x==tail_bloc_y*tail_bloc_x):
                max_tail_x=0
            else:
                max_tail_x= int(GEMM_x/tail_bloc_y*tail_bloc_x)    
        else:
            if(GEMM_y==tail_bloc_y):
                max_tail_y=0
            else:
                max_tail_y= int(GEMM_y/tail_bloc_y)
            if(GEMM_x==tail_bloc_x):
                max_tail_x=0
            else:
                max_tail_x= int(GEMM_x/tail_bloc_x)   

        tot_num_blocks=(max_tail_y+1)*(max_tail_x+1)
        tot_neurons_per_block=tail_bloc_y*tail_bloc_x

        max_num_faulty_neurons=int(neuron_fault_rate*tot_neurons_per_block)
        max_num_faulty_blocks=int(tot_num_blocks*block_error_rate)

        if (max_num_faulty_blocks==0):
            max_num_faulty_blocks = 1

        BlockID_x=[]
        BlockID_y=[]
        tmp_val=[]
        
        fault_info[layer]={'layer':layer,
                           'tot_blocks':tot_num_blocks,
                           'faulty_blocks':max_num_faulty_blocks,
                           'faulty_neuron':max_num_faulty_neurons}

        while(max_num_faulty_blocks!=0):
            block_rnd=random.randint(0,tot_num_blocks)
            if block_rnd not in tmp_val:
                tmp_val.append(block_rnd)
                BlockID_x.append(int(block_rnd%(max_tail_x+1)))
                BlockID_y.append(int(block_rnd/(max_tail_x+1)))
                max_num_faulty_blocks-=1
        
        Neuron_x=[]
        Neuron_y=[]
        tmp_val=[]
        while(max_num_faulty_neurons!=0):
            neuron_rnd = random.randint(0,tot_neurons_per_block)
            if neuron_rnd not in tmp_val:
                tmp_val.append(neuron_rnd)
                Neuron_x.append(int(neuron_rnd%(tail_bloc_x+1)))
                Neuron_y.append(int(neuron_rnd/(tail_bloc_x+1)))
                max_num_faulty_neurons-=1

        #locations=([loc_neuron(layer=layer,dim=dim,shape=shape,BlockID_y=BlockID_y[blk_idx],BlockID_x=BlockID_x[blk_idx],tail_bloc_x=tail_bloc_x,tail_bloc_y=tail_bloc_y) for blk_idx in range(len(BlockID_x)) for _ in range(max_num_faulty_neurons)] * pfi_model.batch_size)
        
        batch_orderx=[0 for _ in range(len(Neuron_x)*len(BlockID_x))]
        
        
        for blokid in range(len(BlockID_x)):
            for neuronid in range(len(Neuron_x)):
                locations.append(loc_neuron(layer=layer,dim=dim,shape=shape,
                                            BlockID_y=BlockID_y[blokid],BlockID_x=BlockID_x[blokid],
                                            Neuron_x=Neuron_x[neuronid],Neuron_y=Neuron_y[neuronid],
                                            tail_bloc_x=tail_bloc_x,tail_bloc_y=tail_bloc_y))
        batch_order.extend(batch_orderx)
    return (locations, batch_order, fault_info)


    


class FI_report_classifier(object):
    def __init__(self,log_pah,chpt_file="chpt_file.json",fault_report_name="fsim_report.csv",) -> None:
        self.chpt_file_name=chpt_file
        self.fault_report_filename=fault_report_name
        self._k=0
        self._c_in=0
        self._kH=0
        self._kW=0
        self._layer=0    
        self._num_imgs = 0
        self._gold_iou=torch.tensor([0.0])
        self._faul_iou=torch.tensor([0.0])
        
        # golden map
        self.g_map = torch.tensor([0.0])
        self.g_map = torch.tensor([0.0])
        self.g_map_50 = torch.tensor([0.0])
        self.g_map_75 = torch.tensor([0.0])
        self.g_map_small = torch.tensor([0.0])
        self.g_map_medium = torch.tensor([0.0])
        self.g_map_large = torch.tensor([0.0])
        self.g_mar_1  = torch.tensor([0.0])
        self.g_mar_10 = torch.tensor([0.0])
        self.g_mar_100 = torch.tensor([0.0])
        self.g_mar_small = torch.tensor([0.0])
        self.g_mar_medium = torch.tensor([0.0])
        self.g_mar_large = torch.tensor([0.0])
        # self.g_map_per_class = torch.tensor([0.0])
        # self.g_mar_100_per_class = torch.tensor([0.0])

        # faulty map
        self.f_map = torch.tensor([0.0])
        self.f_map_50 = torch.tensor([0.0])
        self.f_map_75 = torch.tensor([0.0])
        self.f_map_small = torch.tensor([0.0])
        self.f_map_medium = torch.tensor([0.0])
        self.f_map_large = torch.tensor([0.0])
        self.f_mar_1 = torch.tensor([0.0])
        self.f_mar_10 = torch.tensor([0.0])
        self.f_mar_100 = torch.tensor([0.0])
        self.f_mar_small = torch.tensor([0.0])
        self.f_mar_medium = torch.tensor([0.0])
        # self.f_mar_large = torch.tensor([0.0])
        # self.f_map_per_class = torch.tensor([0.0])
        self.f_mar_100_per_class = torch.tensor([0.0])
        
        self._report_dictionary={}
        self._FI_results_dictionary={}
        self._fault_dictionary={}        
        self.Top1_faulty_code=0
        self.SDC=0
        self.Masked=0
        self.Critical=0

        self.Golden={}
        self.SDC=0
        self.Critical=0
        self.Masked=0
        self.Giou=0
        self.Fiou=0

        self.Giou=torch.tensor([0.0])
        self.Fiou=torch.tensor([0.0])
        self.fiou = torch.tensor([0.0])
        self.giou = torch.tensor([0.0])
        self.Full_report=pd.DataFrame()

        self.ioum_SDC=0
        self.ioum_Critical=0
        self.ioum_Masked=0
        self.log_path=log_pah
        self.report_summary={}
        self._fsim_report=pd.DataFrame()
        self.check_point={
            "fault_idx":0,
            "top1": {
                "fault":{
                        "Critical":0,
                        "SDC":0,
                        "Masked":0
                        },
                "images":{
                        "Critical":0,
                        "SDC":0,
                        "Masked":0
                        }
            }
        }

    def set_fault_report(self, data):
        for key in data:
            self._fault_dictionary[key]=data[key]

    def load_check_point(self):
        ckpt_path_file=os.path.join(self.log_path,self.chpt_file_name)
        if not os.path.exists(ckpt_path_file):
            with open(ckpt_path_file,'w') as Golden_file:
                json.dump(self.check_point,Golden_file)
        else:
            with open(ckpt_path_file,'r') as Golden_file:
                chpt=json.load(Golden_file)
                self.check_point=chpt

        if not os.path.exists(os.path.join(self.log_path,self.fault_report_filename)):
            self._fsim_report=pd.DataFrame(columns=['gold_iou@1',
                                        'boxes_Crit','boxes_SDC','boxes_Masked',
                                        'fault_iou@1','Class', 'gold_map','g_map_50','g_map_75','g_map_small',
                                        'g_map_medium','g_map_large','g_mar_1','g_mar_10','g_mar_100','g_mar_small',
                                        'g_mar_medium','g_mar_large',
                                        'fault_map','f_map_50','f_map_75','f_map_small',
                                        'f_map_medium','f_map_large','f_mar_1','f_mar_10','f_mar_100','f_mar_small',
                                        'f_mar_medium','f_mar_large'])  
            self._fsim_report.to_csv(os.path.join(self.log_path,self.fault_report_filename),sep=',')
        else:
            self._fsim_report = pd.read_csv(os.path.join(self.log_path,self.fault_report_filename),index_col=[0])           
        

    def update_check_point(self):  
        self._update_chpt_info()
        self.update_fault_parse_results()
        self.reset_counter()
        # fidx=self.check_point["fault_idx"]
        # Report_name=f"FI_{fidx}_results.json"
        # FI_report_json_file=os.path.join(self.log_path,Report_name)
        # with open(FI_report_json_file,'w') as Golden_file:
        #     json.dump(self._FI_results_dictionary,Golden_file)        
        # self._FI_results_dictionary={}

        new_row=pd.DataFrame(self._fault_dictionary, index=[0])
        self._fsim_report=pd.concat([self._fsim_report, new_row],ignore_index=True, sort=False)
        self._fsim_report.to_csv(os.path.join(self.log_path,'fsim_report.csv'),sep=',')  
              
        ckpt_path_file=os.path.join(self.log_path,self.chpt_file_name)
        with open(ckpt_path_file,'w') as Golden_file:
            json.dump(self.check_point,Golden_file)


    def reset_counter(self):
        
        # self.mAP=torch.tensor([0.0])
        # self.iou_per_box = torch.tensor([0.0])
        self.SDC=0
        self.Masked=0
        self.Critical=0
        self._num_imgs = 0

        self.Giou = torch.tensor([0.0])
        self.giou = torch.tensor([0.0])
        self.g_map = torch.tensor([0.0])
        self.g_map_50 = torch.tensor([0.0])
        self.g_map_75 = torch.tensor([0.0])
        self.g_map_small = torch.tensor([0.0])
        self.g_map_medium = torch.tensor([0.0])
        self.g_map_large = torch.tensor([0.0])
        self.g_mar_1 = torch.tensor([0.0])
        self.g_mar_10 = torch.tensor([0.0])
        self.g_mar_100 = torch.tensor([0.0])
        self.g_mar_small = torch.tensor([0.0])
        self.g_mar_medium = torch.tensor([0.0])
        self.g_mar_large = torch.tensor([0.0])
        # self.g_map_per_class = torch.tensor([0.0])
        # self.g_mar_100_per_class = torch.tensor([0.0])

        self.Fiou = torch.tensor([0.0])
        self.fiou = torch.tensor([0.0])
        self.f_map = torch.tensor([0.0])
        self.f_map_50 = torch.tensor([0.0])
        self.f_map_75 = torch.tensor([0.0])
        self.f_map_small = torch.tensor([0.0])
        self.f_map_medium = torch.tensor([0.0])
        self.f_map_large = torch.tensor([0.0])
        self.f_mar_1 = torch.tensor([0.0])
        self.f_mar_10 = torch.tensor([0.0])
        self.f_mar_100 = torch.tensor([0.0])
        self.f_mar_small = torch.tensor([0.0])
        self.f_mar_medium = torch.tensor([0.0])
        self.f_mar_large = torch.tensor([0.0])
        # self.f_map_per_class = torch.tensor([0.0])
        # self.f_mar_100_per_class = torch.tensor([0.0])

    def _update_chpt_info(self):    
        self.Top1_faulty_code=0 # 0: Masked; 1: SDC; 2; Critical; 3=crash
        self.Topk_faulty_code=0 # 0: Masked; 1: SDC; 2; Critical; 3=crash
        self.check_point["fault_idx"]+=1 

        self.check_point["top1"]["images"]["Critical"]+=self.Critical
        self.check_point["top1"]["images"]["SDC"]+=self.SDC
        self.check_point["top1"]["images"]["Masked"]+=self.Masked

            # break
        if(self.Critical!=0):
            # self.Critical+=1
            self.check_point["top1"]["fault"]["Critical"]+=1
            self.Top1_faulty_code=2
        elif self.SDC !=0:
            # self.SDC+=1
            self.check_point["top1"]["fault"]["SDC"]+=1
            self.Top1_faulty_code=1
        else:
            # self.Masked+=1
            self.check_point["top1"]["fault"]["Masked"]+=1
            self.Top1_faulty_code=0
        
        print(f'self._num_imgs: {self._num_imgs}')
        self.Giou = (self.giou / self._num_imgs) * 100
        self.Fiou = (self.fiou / self._num_imgs) * 100
    
        # self.Giou=self._gold_iou*100/self._num_images
        # self.Fiou=self._faul_iou*100/self._num_images


    def update_fault_parse_results(self):
        # if isinstance(self.Giou, torch.Tensor):
        self._fault_dictionary['gold_iou@1'] = self.Giou.item()
        # else:
        #     self._fault_dictionary['gold_iou@1'] = self.Giou.item()
        self._fault_dictionary['boxes_Crit'] = self.Critical
        self._fault_dictionary['boxes_SDC'] = self.SDC
        self._fault_dictionary['boxes_Masked'] = self.Masked
        self._fault_dictionary['fault_iou@1'] = self.Fiou.item()
        self._fault_dictionary['Class'] = self.Top1_faulty_code

        # golden map
        self._fault_dictionary['gold_map'] = self.g_map.item()
        self._fault_dictionary['g_map_50'] = self.g_map_50.item()
        self._fault_dictionary['g_map_75'] = self.g_map_75.item()
        self._fault_dictionary['g_map_small'] = self.g_map_small.item()
        self._fault_dictionary['g_map_medium'] = self.g_map_medium.item()
        self._fault_dictionary['g_map_large'] = self.g_map_large.item()
        self._fault_dictionary['g_mar_1'] = self.g_mar_1.item()
        self._fault_dictionary['g_mar_10'] = self.g_mar_10.item()
        self._fault_dictionary['g_mar_100'] = self.g_mar_100.item()
        self._fault_dictionary['g_mar_small'] = self.g_mar_small.item()
        self._fault_dictionary['g_mar_medium'] = self.g_mar_medium.item()
        self._fault_dictionary['g_mar_large'] = self.g_mar_large.item()
        # self._fault_dictionary['g_map_per_class'] = self.g_map_per_class.item()
        # self._fault_dictionary['g_mar_100_per_class'] = self.g_mar_100_per_class.item()

        # fault map
        self._fault_dictionary['fault_map'] = self.f_map.item()
        self._fault_dictionary['f_map_50'] = self.f_map_50.item()
        self._fault_dictionary['f_map_75'] = self.f_map_75.item()
        self._fault_dictionary['f_map_small'] = self.f_map_small.item()
        self._fault_dictionary['f_map_medium'] = self.f_map_medium.item()
        self._fault_dictionary['f_map_large'] = self.f_map_large.item()
        self._fault_dictionary['f_mar_1'] = self.f_mar_1.item()
        self._fault_dictionary['f_mar_10'] = self.f_mar_10.item()
        self._fault_dictionary['f_mar_100'] = self.f_mar_100.item()
        self._fault_dictionary['f_mar_small'] = self.f_mar_small.item()
        self._fault_dictionary['f_mar_medium'] = self.f_mar_medium.item()
        self._fault_dictionary['f_mar_large'] = self.f_mar_large.item()
        # self._fault_dictionary['f_map_per_class'] = self.f_map_per_class.item()
        # self._fault_dictionary['f_mar_100_per_class'] = self.f_mar_100_per_class.item()

        # print(f'self._fault_dictionary: {self._fault_dictionary}')

    



    def create_report(self,file_name):
        listdirectories=os.listdir(self.log_path)
        num=0 
        file_name_cmp=f"{file_name}_"   
        for dir in listdirectories:      
            if file_name_cmp in dir:            
                num+=1
        num+=1
        new_golden_name=f"{file_name}_{num}.json"
        old_report_name=f"{file_name}.json"
        if os.path.exists(os.path.join(self.log_path,old_report_name)):
            os.system(f"mv {os.path.join(self.log_path,old_report_name)} {os.path.join(self.log_path,new_golden_name)}")
        self._report_dictionary=self.load_report(file_name)
        

    def load_report(self,file_name):

        file_name_json=f"{file_name}.json"
        golden_file_path=os.path.join(self.log_path,file_name_json)
        loaded_report={}
        if not os.path.exists(golden_file_path):
            with open(golden_file_path,'w') as Golden_file:
                json.dump(loaded_report,Golden_file)
        else:            
            with open(golden_file_path,'r') as Golden_file:
                loaded_report=json.load(Golden_file)
        return(loaded_report)

    def save_report(self,file_name):    
        file_name_json=f"{file_name}.json"    
        golden_file_path=os.path.join(self.log_path,file_name_json)
        with open(golden_file_path,'w') as Golden_file:
            json.dump(self._report_dictionary,Golden_file)
    
    def update_detection_report(self,index,outputs,targets):
        self._report_dictionary[index]={}
        # outputs: {boxes, labels, scores}
        self._report_dictionary[index]['pred_boxes']=outputs[0]['boxes'].tolist()
        self._report_dictionary[index]['pred_labels']=outputs[0]['labels'].tolist()
        self._report_dictionary[index]['gt_boxes']=targets[0]['boxes'].to('cpu').tolist()
        self._report_dictionary[index]['gt_labels']=targets[0]['labels'].to('cpu').tolist()
        self._report_dictionary[index]['scores']=outputs[0]['scores'].tolist()

        
        
    def Fault_parser(self,golden_file_report, faulty_file_report):
        self._golden_dictionary=self.load_report(golden_file_report)
        self._FI_dictionary=self.load_report(faulty_file_report)
        self.Full_report = pd.DataFrame()
        for index in self._golden_dictionary:
            self.Golden=self._golden_dictionary[index]
            G_pred_bb=torch.tensor(self.Golden['pred_boxes'],requires_grad=False)
            G_pred_labels=torch.tensor(self.Golden['pred_labels'],requires_grad=False)
            G_pred_scores=torch.tensor(self.Golden['scores'], requires_grad=False)
            G_gt_bb=torch.tensor(self.Golden['gt_boxes'],requires_grad=False)
            G_gt_labels=torch.tensor(self.Golden['gt_labels'],requires_grad=False)

            f_buffer_per_img = 0.0
            g_buffer_per_img = 0.0
            # print(f'G_gt_labels: {G_gt_labels}')
            
            if G_gt_labels.nelement() != 0:
                # print(f'G_gt_labels: {G_gt_labels}')
                golden_pred_dict, gt_dict = setup_dicts(pred_labels=G_pred_labels, pred_scores=G_pred_scores, pred_bb=G_pred_bb, gt_labels=G_gt_labels, _gt_bbs=G_gt_bb)

                
                if index in self._FI_dictionary:
                    self._num_imgs += 1
                    # print(f'index: {index}')
                    self.Faulty = self._FI_dictionary[index]

                    F_pred_bb=torch.tensor(self.Faulty['pred_boxes'],requires_grad=False)
                    F_pred_labels=torch.tensor(self.Faulty['pred_labels'],requires_grad=False)
                    F_pred_scores=torch.tensor(self.Faulty['scores'], requires_grad=False)
                    F_gt_bb=torch.tensor(self.Faulty['gt_boxes'],requires_grad=False)
                    F_gt_labels=torch.tensor(self.Faulty['gt_labels'],requires_grad=False)


                    # this function must be improved
                    # relative_faulty_iou(F_gt_labels, F_gt_bb, F_pred_labels, F_pred_bb, F_pred_scores, golden_pred_dict)
                    faulty_dict, gt_dict = setup_dicts(pred_labels=F_pred_labels, pred_scores=F_pred_scores, pred_bb=F_pred_bb, gt_labels=F_gt_labels, _gt_bbs=F_gt_bb)
                    # print(f'fault_dict: {faulty_dict}')
                    # print(f'gt_dict: {gt_dict}')
                    gt_labels_array = np.array(list(gt_dict.keys()))

                    golden_score_per_label = list()

                    # for each golden label
                    # print(f'golden_pred_dict.keys(): {golden_pred_dict.keys()}')
                    for G_idx, (G_label, bbs_scores) in enumerate(golden_pred_dict.items()):
                        # for each target label
                        result = defaultdict(lambda:[])
                        t_position = np.where(G_label == gt_labels_array)
                        t_labels = gt_labels_array[t_position]
                        if len(t_labels) > 0:
                            t_label = t_labels[0]
                            # print(f't_label: {t_label}')
                            # t_label = gt_labels_array[t_label_idx]
                            # for each faulty label
                            buffer_faulty_per_label = 0
                            for F_idx, F_label in enumerate(list(faulty_dict.keys())):
                                # self._num_faulty_images += 1
                                # if faulty label is the right
                                # print(f'F_label == t_label? {F_label == t_label}')
                                if F_label == t_label:
                                    # print(F_label)
                                    # compute the iou with respect to the golden prediction
                                    buffer_faulty_score = 0
                                    buffer_conf = 0
                                    buffer_g_conf = 0
                                    fault_bbs = np.array([arr[0] for arr in faulty_dict[F_label]])
                                    scores = [score[1] for score in faulty_dict[F_label]]
                                    for bb_score in bbs_scores:
                                        bb = bb_score[0]
                                        g_score = bb_score[1]
                                        faulty_disatnces1 = np.linalg.norm(bb[0:2] - fault_bbs[:,0:2], axis=1, ord=2)
                                        faulty_disatnces2 = np.linalg.norm(bb[2:4] - fault_bbs[:,2:4], axis=1, ord=2)

                                        f_buffer = faulty_disatnces1 + faulty_disatnces2

                                        # take the lowest one
                                        f_candidate_idx = np.argmin(f_buffer)

                                        # take the array correspinding to the lowest distance from the reference gt_bb 
                                        f_candidate_bb = fault_bbs[f_candidate_idx]
                                        buffer_conf += scores[f_candidate_idx]
                                        buffer_g_conf += g_score
                                        # compute the score between the nearest bb and the gt_bb
                                        f_score = compute_iou(bb, f_candidate_bb)
                                        # print(f'f_score: {f_score}')
                                        # save result
                                        # score_per_label.append((F_label, score))
                                        buffer_faulty_score += f_score

                                        fault_bbs = np.delete(fault_bbs, f_candidate_idx, axis = 0)
                                        if len(fault_bbs) == 0:
                                            break
                                    normalized_conf = buffer_conf/len(bbs_scores)
                                    normalized_faulty = buffer_faulty_score/len(bbs_scores)
                                    normaized_g_conf =  buffer_g_conf / len(bbs_scores)
                                    # print(f'normalized_faulty: {normalized_faulty}')
                                    if normalized_faulty == 1:
                                        coverage = 'masked'
                                        result[t_label].append((coverage, normalized_faulty))
                                        self.Masked += 1

                                    elif normalized_faulty < 1 and normalized_faulty > 0.6:
                                        coverage = 'SDC'
                                        result[t_label].append((coverage, normalized_faulty))
                                        self.SDC += 1
                                    else:
                                        coverage = 'critical'
                                        result[t_label].append((coverage, normalized_faulty))
                                        self.Critical += 1

                                    # pred_bbs[candidate_idx] = np.array([np.nan, np.nan, np.nan, np.nan])
                                    
                                    # print(f'buffer_faulty_per_label: {buffer_faulty_per_label}')
                                else: 
                                    coverage = 'critical'
                                    result[t_label].append((coverage, 0.0))
                                    self.Critical += 1
                                    normalized_faulty = 0.0
                                    normalized_conf = None
                                    normaized_g_conf = None
                                # print(f'normalized_faulty: {normalized_faulty}')
                                buffer_faulty_per_label += normalized_faulty

                                FaultID=faulty_file_report.split("/")[-1].split(".")[0]

                                df = pd.DataFrame({'FaultID':FaultID,
                                                    'imID': index,                                    
                                                    'G_lab':G_label,
                                                    'Pred_idx': F_idx, # for each label predicted by the faulty model                                      
                                                    'F_lab':F_label,
                                                    'G_Target':t_label,
                                                    'f_conf_score': normalized_conf,
                                                    'g_conf_score': normaized_g_conf},index=[0])  
                                self.Full_report = pd.concat([self.Full_report,df],ignore_index=True)
                                
                            if len(list(faulty_dict.keys())) > 0:
                                f_buffer_per_img += buffer_faulty_per_label / len(list(faulty_dict.keys()))
                            else: 
                                f_buffer_per_img += 0.0
                                    

                            buffer_golden_score = 0
                            golden_pred_bbs = np.array([arr[0] for arr in golden_pred_dict[G_label]])
                            gt_bbs = gt_dict[t_label]

                            for gt_bb in gt_bbs:
                                golden_disatnces1 = np.linalg.norm(gt_bb[0:2] - golden_pred_bbs[:,0:2], axis=1)
                                golden_disatnces2 = np.linalg.norm(gt_bb[2:4] - golden_pred_bbs[:,2:4], axis=1)

                                g_buffer = golden_disatnces1 + golden_disatnces2

                                g_candidate_idx = np.argmin(g_buffer)

                                g_candidate_bb = golden_pred_bbs[g_candidate_idx]

                                g_score = compute_iou(gt_bb, g_candidate_bb)

                                golden_score_per_label.append((G_label, g_score))

                                golden_pred_bbs = np.delete(golden_pred_bbs, g_candidate_idx, axis = 0)

                                buffer_golden_score += g_score

                                if len(golden_pred_bbs) == 0:
                                    break

                        
                            g_buffer_per_img += buffer_golden_score /len(gt_bbs)
                        else:
                            g_buffer_per_img += 0.0

                        gt_labels_array = np.delete(gt_labels_array, t_position)

                    if len(list(faulty_dict.keys())) > 0:
                        self.fiou += f_buffer_per_img / len(list(faulty_dict.keys()))
                    else: 
                        self.fiou += 0.0

                    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
            
                    F_map = compute_mAP(metric_setting=metric, gt_labels=F_gt_labels, gt_bb= F_gt_bb, pred_labels=F_pred_labels, pred_bb=F_pred_bb, pred_scores=F_pred_scores)
                    # print(f'F_map: {F_map}')
                    self.f_map = F_map['map']
                    self.f_map_50 = F_map['map_50']
                    self.f_map_75 = F_map['map_75']
                    self.f_map_small = F_map['map_small']
                    self.f_map_medium = F_map['map_medium']
                    self.f_map_large = F_map['map_large']
                    self.f_mar_1 = F_map['mar_1']
                    self.f_mar_10 = F_map['mar_10']
                    self.f_mar_100 = F_map['mar_100']
                    self.f_mar_small = F_map['mar_small']
                    self.f_mar_medium = F_map['mar_medium']
                    self.f_mar_large = F_map['mar_large']
                    # self.f_map_per_class = F_map['map_per_class']
                    # self.f_mar_100_per_class = F_map['mar_100_per_class']
                
                if len(list(golden_pred_dict.keys())) > 0:
                    self.giou += g_buffer_per_img / len(list(golden_pred_dict.keys()))
                else: 
                    self.giou += 0.0
                

                metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    
                G_map = compute_mAP(metric_setting=metric, gt_labels=G_gt_labels, gt_bb= G_gt_bb, pred_labels=G_pred_labels, pred_bb=G_pred_bb, pred_scores=G_pred_scores)
                # print(f'G_map: {G_map}')
                self.g_map = G_map['map']
                self.g_map_50 = G_map['map_50'] # 50 is the confidence level of the model
                self.g_map_75 = G_map['map_75'] # 75 is the confidence level of the model
                self.g_map_small = G_map['map_small']
                self.g_map_medium = G_map['map_medium']
                self.g_map_large = G_map['map_large']
                self.g_mar_1 = G_map['mar_1']
                self.g_mar_10 = G_map['mar_10']
                self.g_mar_100 = G_map['mar_100']
                self.g_mar_small = G_map['mar_small']
                self.g_mar_medium = G_map['mar_medium']
                self.g_mar_large = G_map['mar_large']
                # self.g_map_per_class = G_map['map_per_class']
                # self.g_mar_100_per_class = G_map['mar_100_per_class']

        file_name=faulty_file_report.split('/')[-1].split('.')[0]
        csv_report=f"{file_name}.csv"
        if(len(self.Full_report)>0):
            self.Full_report.to_csv(os.path.join(self.log_path,csv_report))
           
        self._report_dictionary=self._FI_dictionary
        self._FI_dictionary={}
        self._golden_dictionary={}
        
        #self.save_report(faulty_file_report)
            #return(FI_results_json)




class FI_framework(object):
    def __init__(self,log_path,mode='single_fault') -> None:
        # self.fault_dictionary={}   
        self._BER=0 
        self._layer=[]
        self._kK=[]
        self._kC=[]
        self._kH=[]
        self._kW=[]                
        self._inj_mask=[]         
        self._bf_inj_w_mask=0
        self.faultfreedata=0
        self.faultydata=0
        self.log_msg=''
        self.log_path=log_path
        self.log_file=os.path.join(log_path,'FSIM_log.log')
        self.pfi_model=None
        self.injected_fault={}
        
    def float_to_hex(self,f):
        h=hex(struct.unpack('<I', struct.pack('<f', f))[0])
        return h[2:len(h)]

    def hex_to_float(self,h):
        return float(struct.unpack(">f",struct.pack(">I",int(h,16)))[0])

    def int_to_float(self,h):
        return float(struct.unpack(">f",struct.pack(">I",h))[0])

    def create_fault_injection_model(self,device,model,batch_size=1,input_shape=[3,224,224],layer_types=[torch.nn.Conv2d],Neurons=False): 
        if device.type.startswith('cuda'): 
            use_cuda=True
        else:
            use_cuda=False
        if Neurons:
            self.pfi_model = single_bit_flip_func(model, 
                        batch_size=batch_size,
                        input_shape=input_shape,
                        layer_types=layer_types,
                        use_cuda=use_cuda,
                        bits=8,
                        )
        else:
            self.pfi_model = FaultInjection(model, 
                        batch_size=batch_size,
                        input_shape=input_shape,
                        layer_types=layer_types,
                        use_cuda=use_cuda,
                        )
        # self.pfi_model.print_pytorchfi_layer_summary()
    
    def bit_flip_err_neuron(self,fault):
        
        print(fault[0])
        layer_start=fault[0]['layer_start']
        layer_stop=fault[0]['layer_stop']
        block_fault_rate=fault[0]['block_fault_rate']
        neuron_fault_rate=fault[0]['neuron_fault_rate']
        size_tail_y=fault[0]['size_tail_y']
        size_tail_x=fault[0]['size_tail_x']
        bit_faulty_pos=fault[0]['bit_faulty_pos']


        #locations=([generate_error_list_neurons(self.pfi_model,layer=layer) for _ in range(berr)] * self.pfi_model.batch_size)
        #batch_order=[i for i in range(self.pfi_model.batch_size)]*berr        
        (locations,batch_order,fault_info)=generate_error_list_neurons_tails(self.pfi_model,
                                                                  layer_i=layer_start,
                                                                  layer_n=layer_stop,
                                                                  block_error_rate=block_fault_rate,
                                                                  neuron_fault_rate=neuron_fault_rate,
                                                                  tail_bloc_y=size_tail_y,
                                                                  tail_bloc_x=size_tail_x)        
        
        #logger.info()

        # this weird list is andatory for the original fasult injector 
        #self.pfi_model.set_conv_max([255.0 for _ in range(self.pfi_model.get_total_layers())])

        # here I changed it in order to inject bit-flips, more representative
        self.pfi_model.set_conv_max([bit_faulty_pos])

        #print(locations)
        #(layer, C, H, W) = random_neuron_location(self.pfi_model)
        random_layers, random_c, random_h, random_w = map(list, zip(*locations))
        print(f"lengts={len(random_layers)} {len(batch_order)} {len(random_c)} {len(random_h)} {len(random_w)}")
        #print(batch_order, random_layers, random_c, random_h, random_w)
        self.faulty_model = self.pfi_model.declare_neuron_fault_injection(
            layer_num=random_layers,
            batch=batch_order,
            dim1=random_c,
            dim2=random_h,
            dim3=random_w,
            function=self.pfi_model.single_bit_flip_across_batch_tensor,
        )
        for key in fault_info:
            self.log_msg=f"Fault=layer:{fault_info[key]['layer']}, block_rate:{block_fault_rate}, neuron_rate:{neuron_fault_rate}, tot_blocks:{fault_info[key]['tot_blocks']}, faulty_blocks:{fault_info[key]['faulty_blocks']}, faulty_neuron:{fault_info[key]['faulty_neuron']}, bit_loc:{bit_faulty_pos}, "
            logger.info(self.log_msg)
        self.log_msg=""
        self.faulty_model.eval()

    def bit_flip_weight_inj(self, fault):
        layer=[fault[0]['layer']]
        k=[fault[0]['kernel']]
        c_in=[fault[0]['channel']]
        kH=[fault[0]['row']]
        kW=[fault[0]['col']]
        inj_mask=[fault[0]['bitmask']]

        self._kK=(k)
        self._kC=(c_in)
        self._kH=(kH)
        self._kW=(kW)  
        self._inj_mask=(inj_mask)
        self._layer=layer
        if k!=None or c_in!=None:
            self.faulty_model=self.pfi_model.declare_weight_fault_injection(
                BitFlip=self._bit_flip_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW, bitmask=inj_mask
            )
        else:
            self.faulty_model=self.pfi_model.declare_weight_fault_injection(
                BitFlip=self._bit_flip_weight, layer_num=layer, k=k, dim1=c_in, bitmask=inj_mask
            )

        self.faulty_model.eval()

    def _bit_flip_weight(self,data, location, injmask):
        orig_data=data[location].item()
        data_32bit=int(self.float_to_hex(data[location].item()),16)

        corrupt_32bit=data_32bit ^ int(injmask)
        corrupt_val=self.int_to_float(corrupt_32bit)
        self.log_msg=f"F_descriptor: Layer:{self._layer}, (K, C, H, W):{location}, BitMask:{injmask}, Ffree_Weight:{data_32bit}, Faulty_weight:{corrupt_32bit}"
        logger.info(self.log_msg)
        fsim_dict={'Layer':self._layer[0], 
                    'kernel':self._kK[0],
                    'channel':self._kC[0],
                    'row':self._kH[0],
                    'col':self._kW[0],
                    'BitMask':self._inj_mask[0],
                    'Ffree_Weight':data_32bit,
                    'Faulty_weight':corrupt_32bit,
                    'Abs_error':(orig_data-corrupt_val)}
        self.injected_fault=fsim_dict
        return corrupt_val
    

    def BER_weight_inj(self, BER, layer=None, kK=None, kC=None, kH=None, kW=None, inj_mask=None):       
        self._layer=[]
        self._kK=[]
        self._kC=[]
        self._kH=[]
        self._kW=[]                
        self._inj_mask=[]
        N_layers=1
        N_Kernels=1
        N_Channels=1
        N_Rows=1
        N_Cols=1
        N_Bits=32              
        Tot_N_bits=0
        Bit_mask_selected=1
        fsim_dict={}
        err_list=[]
        num_errors=0
        while(num_errors<BER):
            if layer:
                N_layers=1
                layer_selected=layer-1
                fsim_dict['layer']=layer_selected
            else:
                N_layers=self.pfi_model.get_total_layers()
                layer_selected=random.randint(0,N_layers-1)                    
            weight_shape=list(self.pfi_model.get_weights_size(layer_selected))
            if kK:
                N_Kernels=1                
                Kernel_selected=kK-1
                fsim_dict['kernel']=Kernel_selected
            else:
                N_Kernels=weight_shape[0]
                Kernel_selected=random.randint(0,N_Kernels-1)

            if kC:
                N_Channels=1                
                Channel_selected=kC-1
                fsim_dict['channel']=Channel_selected
            else:
                N_Channels=weight_shape[1]
                Channel_selected=random.randint(0,N_Channels-1)
            if kH:
                N_Rows=1                
                Row_selected=kH-1
                fsim_dict['row']=Row_selected
            else:
                N_Rows=weight_shape[2]
                Row_selected=random.randint(0,N_Rows-1)
            if kW:
                N_Cols=1                
                Col_selected=kH-1
                fsim_dict['col']=Col_selected
            else:
                N_Cols=weight_shape[3]
                Col_selected=random.randint(0,N_Cols-1)

            Bit_mask_selected=2**(random.randint(0,31))

            tmp=[layer_selected,Kernel_selected,Channel_selected,Row_selected,Col_selected,Bit_mask_selected]
            if tmp not in err_list:
                err_list.append(tmp)
                self._layer.append(layer_selected)
                self._kK.append(Kernel_selected)
                self._kC.append(Channel_selected)
                self._kH.append(Row_selected)
                self._kW.append(Col_selected)
                self._inj_mask.append(Bit_mask_selected)
                num_errors+=1                    

        Tot_N_bits=N_layers*N_Kernels*N_Channels*N_Rows*N_Cols*N_Bits
        self._BER=BER/Tot_N_bits
        fsim_dict['N_BER']=BER
        fsim_dict['T_Bits']=Tot_N_bits
        fsim_dict['BER']=self._BER
        self.injected_fault=fsim_dict
        
        self.faulty_model=self.pfi_model.declare_weight_fault_injection(
            BitFlip=self._BER_weight, layer_num=self._layer, k=self._kK, dim1=self._kC, dim2=self._kH, dim3=self._kW, bitmask=self._inj_mask
        )
        self.faulty_model.eval()


    def _BER_weight(self,data, location, injmask):
        orig_data=data[location].item()
        data_32bit=int(self.float_to_hex(data[location].item()),16)
        corrupt_32bit=data_32bit ^ int(injmask)
        corrupt_val=self.int_to_float(corrupt_32bit)

        return corrupt_val




class DatasetSampling(object):
    def __init__(self,dataset,num_images):        
        self.length=len(dataset)
        self.num_images=num_images
        self.indices=[]

    def listindex(self):
        self.indices=[]
        for i in range(0,self.length):
            if(i%50)<self.num_images:
                self.indices.append(i)
        return(self.indices)

    def __len__(self):
        return len(self.indices)
    

class FI_manager(object):
    def __init__(self,log_path,chpt_file_name,fault_report_name) -> None:
        self.faulty_model=None
        self.pfi_model=None
        self.fault_list_type='weight'
        self._fault_list=[]
        self.log_msg=''
        self.log_path=log_path
        self.log_file=os.path.join(log_path,'FSIM_log.log')
        self.FI_report=FI_report_classifier(log_path,chpt_file=chpt_file_name,fault_report_name=fault_report_name)
        self.FI_framework=FI_framework(log_path)
        self._golden_file_name=""
        self._faulty_file_name=""
        try:
            os.makedirs(self.log_path)           
        except:
            print(f"The log path: {self.log_path} // already exist...")   


    def generate_fault_list(self, **kwargs): 
        self.pfi_model=self.FI_framework.pfi_model       
        if kwargs:
            if (kwargs.get('flist_mode')=='sbfm'):
                self._fault_list=generate_fault_list_sbfm(self.log_path,self.pfi_model,**kwargs)
            elif(kwargs.get('flist_mode')=='ber'):
                self._fault_list=generate_fault_list_ber(self.log_path,self.pfi_model,**kwargs)

            elif(kwargs.get('flist_mode')=='neurons'):
                self._fault_list=generate_fault_neurons_tailing(self.log_path,self.pfi_model,**kwargs)

            else:
                raise ValueError("The fault list can't be generated in this configuration")
        else:
            raise ValueError("The input arguments are wrong, please be sure you selected at least the fault list name in csv format")
        
    def iter_fault_list(self):
        for k in range(int(self.FI_report.check_point["fault_idx"]),len(self._fault_list)):
            fault_info=self._fault_list.iloc[[k]]
            fault=fault_info.to_dict('records')
            yield (fault, k)    

    def write_reports(self):
        self.FI_report.set_fault_report(self.FI_framework.injected_fault)
        self.FI_report.update_check_point()
        file_name=self._faulty_file_name.split('/')[-1].split('.')[0]
        csv_report=f"{file_name}.csv"
        self.FI_report.Full_report.to_csv(os.path.join(self.log_path,csv_report))
        if os.path.exists(f"{self.log_path}/{self._faulty_file_name}.json"):
            os.remove(f"{self.log_path}/{self._faulty_file_name}.json")
        #self.FI_report.save_report(self._faulty_file_name)

    def load_check_point(self):
        self.FI_report.load_check_point()

    def open_golden_results(self,golden_file_name):
        self._golden_file_name=golden_file_name
        self.FI_report.create_report(golden_file_name)

    def close_golden_results(self):
        self.FI_report.save_report(self._golden_file_name)
        

    def open_faulty_results(self,results_fault_name):
        self._faulty_file_name=results_fault_name
        self.FI_report.create_report(results_fault_name)

    def close_faulty_results(self):
        self.FI_report.save_report(self._faulty_file_name)


    def parse_results(self):    
        self.close_faulty_results()    
        self.FI_report.Fault_parser(self._golden_file_name,self._faulty_file_name)
        self.write_reports()
    
    def terminate_fsim(self):
        pass
    
