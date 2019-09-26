
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[2]:

df = pd.read_csv('./datasets/data_n20_N200.csv',sep=',',header=None)
df.head()


# In[61]:

data_arr = np.array(df.loc[1:,1:],dtype=int)
#data_arr = np.concatenate((data_arr,data_arr))
#data_arr = np.concatenate((data_arr,data_arr))
#data_arr = np.concatenate((data_arr,data_arr))
#data_arr = np.concatenate((data_arr,data_arr))
#data_arr = np.concatenate((data_arr,data_arr))
data_arr.shape


# In[62]:

def generate_V(centre):
    centre_arr = np.array(centre)
    n = len(centre_arr)
    V_order = np.zeros(n)
    distToCentre = V_order
    distToCentre = centre_arr - (np.max(centre_arr)+1)/2
    if(n%2):
        V_order[distToCentre==0] = 1
    for i in distToCentre[distToCentre>0]:
        if (np.random.rand()>0.5):
            V_order[distToCentre == i] = abs(i)*2+1
            V_order[distToCentre == -i] = abs(i)*2
        else:
            V_order[distToCentre == i] = abs(i)*2
            V_order[distToCentre == -i] = abs(i)*2+1
            
    return V_order


# In[63]:

def rank(ls):
    arr = np.array(ls)
    ordering = np.argsort(arr)
    ranks = np.empty_like(ordering)
    ranks[ordering] = np.arange(len(arr))+1
    return ranks


# In[64]:

def pseudo_likelihood(data,n_samples,alpha):
    data_arr = np.array(data)
    N = data_arr.shape[0]
    n = data_arr.shape[1]
    rho_samples = np.zeros((n_samples,n),dtype=int)
    data_rank = rank(np.sum(data_arr,axis=0))
    for i in range(n_samples):
        if((i+1)%100==0):
            print(i+1 ,"/", n_samples, "iterations")
        support = np.arange(0,n)
        rho_tmp = np.zeros(n, dtype = int)
        V_centre = generate_V(data_rank)
        for j in range(n):
            i_curr = np.where(V_centre ==(j+1))
            ####calculating the distance from the data to the possible value
            dists = np.array(list(map(lambda x: np.sum(abs(data_arr[:,i_curr].ravel() - x)),(support+1))))
            log_num =(-alpha/(n)*(dists)) - np.max(-alpha/(n)*(dists))  #minus the max to stablize the exponential term
            log_denom = np.log(np.sum(np.exp(log_num)))
            ###to be very sure that it sums to one, therefore devide by its sum
            probs= np.exp((log_num-log_denom)) / np.sum(np.exp((log_num-log_denom))) 
            index = int(np.where(np.random.multinomial(n=1, pvals = probs)==1)[0])
            rho_tmp[i_curr] = support[index]
            support = support[support!=rho_tmp[i_curr]]
       
        rho_samples[i,:] = rho_tmp+1
           
    return rho_samples


# In[67]:

import time
start = time.time()
rho_samples=pseudo_likelihood(data_arr,2000,1)
end = time.time()
#print(end-start)
print(np.round(end - start,3),"seconds elapsed")


# In[68]:

rho_samples


# In[ ]:



