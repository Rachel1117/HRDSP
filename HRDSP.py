import os
from sklearn.metrics import *
from sklearn.linear_model import *
import math
import numpy as np
import pandas as pd
from math import sqrt
import itertools 
np.random.seed(42)

def row_nor(input):
  rows = input.shape[0]
  cols = input.shape[1]
  output=np.zeros(shape=(rows,cols))
  for i in range(rows) :
    s = sum(input[i])
    for j in range(cols) :
      if s==0:
        output[i,j]==0
      else:
        output[i,j] = input[i,j]/s
  return(output)

def col_nor(input):
  rows = input.shape[0]
  cols = input.shape[1] 
  output=np.zeros(shape=(rows,cols))
  for i in range(rows) :
    s = sum(input[:,i])
    for j in range(cols) :
      if s==0:
        output[i,j]==0
      else:
        output[i,j] = input[i,j]/s
  return(output)

def fix_no_zero(input):
  rows = input.shape[0]
  cols = input.shape[1]
  output=np.zeros(shape=(rows,cols))
  for i in range(rows) :
    for j in range(cols) :
      if input[i,j]==0 and i==j :
        output[i,j] = 0
      elif i==j :
        output[i,j] = input[i,j] ** (-0.5)
      else:
        output[i,j] = input[i,j]
  return(output)

def split_mat(input):
  rows = input.shape[0]
  cols = input.shape[1] 
  mid =  np.sum(input)
  output1 = np.zeros(shape=(rows,mid))
  output2 = np.zeros(shape=(mid,cols))
  start = 0
  for i in range(rows) :
    num = sum(input[i])
    if num != 0 :
      for j in range(start,start+num) :
        output1[i,j] = 1
      start = start + num

  start = 0
  for i in range(cols) :
    num = sum(input[:,i])
    if num != 0:
      for j in range(start,start+num) :
        output2[j,i] = 1
      start = start + num
  output=[output1,output2]
  return(output)
  
def final_nor(input1, input2):
  rows1 = input1.shape[0]
  cols2 = input2.shape[1]
  output = np.zeros(shape=(rows1,cols2))
  for i in range(rows1):
    for j in range(cols2):
      f1 = sum(input1[i]*input2[:,j])
      f2 = np.linalg.norm(input1[i,], axis=None)*np.linalg.norm(input2[:,j], axis=None)
      if f2!=0 :
        output[i,j] = f1/f2
      else :
        output[i,j] = 0
  return(output)

def hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat,m_p):
  drug_num = DD_mat.shape[0]
  comp_mat1 = np.identity(drug_num)   
  comp_mat2 = np.identity(drug_num)    
  if len(m_p)%2==0:#judgment division
    #take the first element
    so = math.floor(len(m_p)/2)-1
    if so==0:
      for j in range(1):
        comp_mat1 = row_nor(comp_mat1)   
      if m_p[j:j+1]=='D' and m_p[j+1:j+2]=='D':
        temp = row_nor(DD_mat)
        #print(j+1)
      elif m_p[j:j+1]=='D' and m_p[j+1:j+2]=='G':
        temp = row_nor(DG_mat)
      elif m_p[j:j+1]=='G' and m_p[j+1:j+2]=='D':
        temp = row_nor(DG_mat.T)
      elif m_p[j:j+1]=='G' and m_p[j+1:j+2]=='G':
        temp = row_nor(GG_mat)
      elif m_p[j:j+1]=='T' and m_p[j+1:j+2]=='T':
        temp = row_nor(TT_mat)
      elif m_p[j:j+1]=='D' and m_p[j+1:j+2]=='T':
        temp = row_nor(DT_mat)
      elif m_p[j:j+1]=='T' and m_p[j+1:j+2]=='D':
        temp = row_nor(DT_mat.T)
      elif m_p[j:j+1]=='G' and m_p[j+1:j+2]=='T':
        temp = row_nor(TG_mat.T)
      elif m_p[j:j+1]=='T' and m_p[j+1:j+2]=='G':
        temp = row_nor(TG_mat)
      comp_mat1 = np.dot(comp_mat1, temp) 

    else:
      for j in range(so):
        comp_mat1 = row_nor(comp_mat1)   
        if m_p[j:j+1]=='D' and m_p[j+1:j+2]=='D':
          temp = row_nor(DD_mat)
          #print(j+1)
        elif m_p[j:j+1]=='D' and m_p[j+1:j+2]=='G':
          temp = row_nor(DG_mat)
        elif m_p[j:j+1]=='G' and m_p[j+1:j+2]=='D':
          temp = row_nor(DG_mat.T)
        elif m_p[j:j+1]=='G' and m_p[j+1:j+2]=='G':
          temp = row_nor(GG_mat)
        elif m_p[j:j+1]=='T' and m_p[j+1:j+2]=='T':
          temp = row_nor(TT_mat)
        elif m_p[j:j+1]=='D' and m_p[j+1:j+2]=='T':
          temp = row_nor(DT_mat)
        elif m_p[j:j+1]=='T' and m_p[j+1:j+2]=='D':
          temp = row_nor(DT_mat.T)
        elif m_p[j:j+1]=='G' and m_p[j+1:j+2]=='T':
          temp = row_nor(TG_mat.T)
        elif m_p[j:j+1]=='T' and m_p[j+1:j+2]=='G':
          temp = row_nor(TG_mat)
        comp_mat1 = np.dot(comp_mat1, temp)  
      
    #first element after taking half
    st = int(len(m_p)/2)-1
    so = len(m_p)-2
    
    if st==0 and so==0:
      for j in range(1):
        comp_mat2 = row_nor(comp_mat2)
        if m_p[j+1:j+2]=='D' and m_p[j:j+1]=='D':
          temp = row_nor(DD_mat)
          #print(j+1)
        elif m_p[j+1:j+2]=='G' and m_p[j:j+1]=='D':
          temp = row_nor(DG_mat.T)
        elif m_p[j+1:j+2]=='D' and m_p[j:j+1]=='G':
          temp = row_nor(DG_mat)
        elif m_p[j+1:j+2]=='G' and m_p[j:j+1]=='G':
          temp = row_nor(GG_mat)
        elif m_p[j+1:j+2]=='T' and m_p[j:j+1]=='T':
          temp = row_nor(TT_mat)
        elif m_p[j+1:j+2]=='T' and m_p[j:j+1]=='D':
          temp = row_nor(DT_mat.T)
        elif m_p[j+1:j+2]=='D' and m_p[j:j+1]=='T':
          temp = row_nor(DT_mat)
        elif m_p[j+1:j+2]=='T' and m_p[j:j+1]=='G':
          temp = row_nor(TG_mat)
        elif m_p[j+1:j+2]=='G' and m_p[j:j+1]=='T':
          temp = row_nor(TG_mat.T)
        comp_mat2 = np.dot(comp_mat2, temp)

    else:
        for j in range(so,st,-1):
          comp_mat2 = row_nor(comp_mat2)
        if m_p[j+1:j+2]=='D' and m_p[j:j+1]=='D':
          temp = row_nor(DD_mat)
          #print(j+1)
        elif m_p[j+1:j+2]=='G' and m_p[j:j+1]=='D':
          temp = row_nor(DG_mat.T)
        elif m_p[j+1:j+2]=='D' and m_p[j:j+1]=='G':
          temp = row_nor(DG_mat)
        elif m_p[j+1:j+2]=='G' and m_p[j:j+1]=='G':
          temp = row_nor(GG_mat)
        elif m_p[j+1:j+2]=='T' and m_p[j:j+1]=='T':
          temp = row_nor(TT_mat)
        elif m_p[j+1:j+2]=='T' and m_p[j:j+1]=='D':
          temp = row_nor(DT_mat.T)
        elif m_p[j+1:j+2]=='D' and m_p[j:j+1]=='T':
          temp = row_nor(DT_mat)
        elif m_p[j+1:j+2]=='T' and m_p[j:j+1]=='G':
          temp = row_nor(TG_mat)
        elif m_p[j+1:j+2]=='G' and m_p[j:j+1]=='T':
          temp = row_nor(TG_mat.T)
        comp_mat2 = np.dot(comp_mat2, temp)   
    
    #center position
    mid_pos = int(len(m_p)/2)-1

    if m_p[mid_pos:mid_pos+1]=='D' and m_p[mid_pos+1:mid_pos+2]=='D':
      AB = split_mat(DD_mat)     
    elif m_p[mid_pos:mid_pos+1]=='D' and m_p[mid_pos+1:mid_pos+2]=='G':
      AB = split_mat(DG_mat)
    elif m_p[mid_pos:mid_pos+1]=='G' and m_p[mid_pos+1:mid_pos+2]=='D':
      AB = split_mat(DG_mat.T)
    elif m_p[mid_pos:mid_pos+1]=='G' and m_p[mid_pos+1:mid_pos+2]=='G':
      AB = split_mat(GG_mat)
    elif m_p[mid_pos:mid_pos+1]=='T' and m_p[mid_pos+1:mid_pos+2]=='T':
      AB = split_mat(TT_mat)
    elif m_p[mid_pos:mid_pos+1]=='D' and m_p[mid_pos+1:mid_pos+2]=='T':
      AB = split_mat(DT_mat)
    elif m_p[mid_pos:mid_pos+1]=='T' and m_p[mid_pos+1:mid_pos+2]=='D':
      AB = split_mat(DT_mat.T)
    elif m_p[mid_pos:mid_pos+1]=='G' and m_p[mid_pos+1:mid_pos+2]=='T':
      AB = split_mat(TG_mat.T)
    elif m_p[mid_pos:mid_pos+1]=='T' and m_p[mid_pos+1:mid_pos+2]=='G':
      AB = split_mat(TG_mat)
    
    A = AB[0]
    B = AB[1]
    
    #print(A.shape)
    #print(B.shape)
    
    comp_mat1 = row_nor(comp_mat1)
    A = row_nor(A)
    comp_mat1 = np.dot(comp_mat1,A)
    comp_mat2 = row_nor(comp_mat2)
    B = B.T
    B = row_nor(B)
    comp_mat2 = np.dot(comp_mat2,B)
    #print(comp_mat1.shape,comp_mat2.shape)
    drugsim_mat = final_nor(comp_mat1,comp_mat2.T)
    
  else:
    #print('odd')
    for j in range(math.floor(len(m_p)/2)):
      comp_mat1 = row_nor(comp_mat1)
      if m_p[j:j+1]=='D' and m_p[j+1:j+2]=='D':
        temp = row_nor(DD_mat)
        #print(1)
      elif m_p[j:j+1]=='D' and m_p[j+1:j+2]=='G':
        temp = row_nor(DG_mat)
        #print(2)
      elif m_p[j:j+1]=='G' and m_p[j+1:j+2]=='D':
        temp = row_nor(DG_mat.T)
        #print(3)
      elif m_p[j:j+1]=='G' and m_p[j+1:j+2]=='G':
        temp = row_nor(GG_mat)
        #print(4)
      elif m_p[j:j+1]=='T' and m_p[j+1:j+2]=='T':
        temp = row_nor(TT_mat)
        #print(5)
      elif m_p[j:j+1]=='D' and m_p[j+1:j+2]=='T':
        temp = row_nor(DT_mat)
        #print(6)
      elif m_p[j:j+1]=='T' and m_p[j+1:j+2]=='D':
        temp = row_nor(DT_mat.T)
        #print(7)
      elif m_p[j:j+1]=='G' and m_p[j+1:j+2]=='T':
        temp = row_nor(TG_mat.T)
        #print(8)
      elif m_p[j:j+1]=='T' and m_p[j+1:j+2]=='G':
        temp = row_nor(TG_mat)
        #print(9)
      comp_mat1 = np.dot(comp_mat1,temp)  
    #print(comp_mat1.shape)
    st = int(len(m_p)/2)-1
    so = len(m_p)-2

    for j in range(so,st,-1):
      comp_mat2 = row_nor(comp_mat2)
      if m_p[j:j+1]=='D' and m_p[j+1:j+2]=='D':
        temp = row_nor(DD_mat)
      elif m_p[j+1:j+2]=='G' and m_p[j:j+1]=='D':
        temp = row_nor(DG_mat.T)
      elif m_p[j+1:j+2]=='D' and m_p[j:j+1]=='G':
        temp = row_nor(DG_mat)
      elif m_p[j+1:j+2]=='G' and m_p[j:j+1]=='G':
        temp = row_nor(GG_mat)
      elif m_p[j+1:j+2]=='T' and m_p[j:j+1]=='T':
        temp = row_nor(TT_mat)
      elif m_p[j+1:j+2]=='T' and m_p[j:j+1]=='D':
        temp = row_nor(DT_mat.T)
      elif m_p[j+1:j+2]=='D' and m_p[j:j+1]=='T':
        temp = row_nor(DT_mat)
      elif m_p[j+1:j+2]=='T' and m_p[j:j+1]=='G':
        temp = row_nor(TG_mat)
      elif m_p[j+1:j+2]=='G' and m_p[j:j+1]=='T':
        temp = row_nor(TG_mat.T)
      comp_mat2 = np.dot(comp_mat2, temp)
    #print(comp_mat1.shape,comp_mat2.shape)
    drugsim_mat = final_nor(comp_mat1,comp_mat2.T)
    #print(drugsim_mat.shape)
  return(drugsim_mat)

def get_feature(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat,DD_sim_ATC1,DD_sim_ATC2,DD_sim_ATC3,DD_sim_chemical,TOT):
    
    
    drug_num = DD_mat.shape[0]    

    SDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DD')
    SDDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DDD')
    SDGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DGD')
    SDCD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DTD')
    SDGDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DGDD')
    SDGGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DGGD')
    SDGTD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DGTD')
    SDDDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DDDD')
    SDDGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DDGD')
    SDDTD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DDTD')
    SDTGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DTGD')
    SDTTD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DTTD')
    SDTDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DTDD')
    SDDDDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DDDDD')
    SDDDGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DDDGD')
    SDDDTD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DDDTD')
    SDDGDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DDGDD')
    SDDGGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DDGGD')
    SDDTDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DDTDD')
    SDDTTD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DDTTD')
    SDGDDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DGDDD')
    SDGDGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DGDGD')
    SDGGDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DGGDD')
    SDGGGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DGGGD')
    SDGTGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DGTGD')
    SDGGTD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DGGTD')
    SDGTTD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DGTTD')
    SDTTTD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DTTTD')
    SDTTDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DTTDD')
    SDTTGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DTTGD')
    SDTGTD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DTGTD')
    SDTGGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DTGGD')
    SDTDGD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DTDGD')
    SDTGDD = hetesim(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat, m_p='DTGDD')
    
    attr_num = 41

    attr = pd.DataFrame(np.zeros(shape=(int(drug_num*(drug_num-1)/2),int(attr_num))))
    pd.options.display.float_format = lambda x : '{:}'.format(x) if round(x,0) == x else '{:,.10f}'.format(x)
    attr_pos = 0
    for m in range(1,drug_num) :
        for n in range(0,m) :
            attr.loc[attr_pos,0] = DD_sim_ATC1[m,n]
            attr.loc[attr_pos,1] = DD_sim_ATC2[m,n]
            attr.loc[attr_pos,2] = DD_sim_ATC3[m,n]
            attr.loc[attr_pos,3] = DD_sim_chemical[m,n]

            attr.loc[attr_pos,4] = SDD[m,n]
            attr.loc[attr_pos,5] = 0.5*SDDD[m,n]
            attr.loc[attr_pos,6] = 0.5*SDGD[m,n] 
            attr.loc[attr_pos,7] = 0.5*SDCD[m,n]

            attr.loc[attr_pos,8] = 0.33*SDGDD[m,n]
            attr.loc[attr_pos,9] = 0.33*SDGGD[m,n]
            attr.loc[attr_pos,10] = 0.33*SDGTD[m,n]
            attr.loc[attr_pos,11] = 0.33*SDDDD[m,n]
            attr.loc[attr_pos,12] = 0.33*SDDGD[m,n]
            attr.loc[attr_pos,13] = 0.33*SDDTD[m,n]
            attr.loc[attr_pos,14] = 0.33*SDTGD[m,n]
            attr.loc[attr_pos,15] = 0.33*SDTTD[m,n]
            attr.loc[attr_pos,16] = 0.33*SDTDD[m,n]

            attr.loc[attr_pos,17] = 0.25*SDDDDD[m,n]
            attr.loc[attr_pos,18] = 0.25*SDDDGD[m,n]
            attr.loc[attr_pos,19] = 0.25*SDDDTD[m,n]
            attr.loc[attr_pos,20] = 0.25*SDDGDD[m,n] 
            attr.loc[attr_pos,21] = 0.25*SDDGGD[m,n] 
            attr.loc[attr_pos,22] = 0.25*SDDTDD[m,n]
            attr.loc[attr_pos,23] = 0.25*SDDTTD[m,n] 

            attr.loc[attr_pos,24] = 0.25*SDGDDD[m,n] 
            attr.loc[attr_pos,25] = 0.25*SDGDGD[m,n] 
            attr.loc[attr_pos,26] = 0.25*SDGGDD[m,n] 
            attr.loc[attr_pos,27] = 0.25*SDGGGD[m,n] 
            attr.loc[attr_pos,28] = 0.25*SDGTGD[m,n] 
            attr.loc[attr_pos,29] = 0.25*SDGGTD[m,n] 
            attr.loc[attr_pos,30] = 0.25*SDGTTD[m,n] 
            
            attr.loc[attr_pos,31] = 0.25*SDTTTD[m,n]
            attr.loc[attr_pos,32] = 0.25*SDTTDD[m,n]
            attr.loc[attr_pos,33] = 0.25*SDTTGD[m,n] 
            attr.loc[attr_pos,34] = 0.25*SDTGTD[m,n]
            attr.loc[attr_pos,35] = 0.25*SDTGGD[m,n]
            attr.loc[attr_pos,36] = 0.25*SDTDGD[m,n]
            attr.loc[attr_pos,37] = 0.25*SDTGDD[m,n] 
            
            attr.loc[attr_pos,40] = DD_mat[m,n]
            attr_pos = attr_pos+1

            #colnames(attr)[attr_num]  c("label")
    attr[38] = attr[[4,38]].mean(axis=1)
    attr[39] = attr[[4,38]].std(ddof=1,axis=1)
    
    attr_pos1= 0
    attr1=pd.DataFrame(np.zeros(shape=(int(drug_num*(drug_num-1)/2),2))) 
    for m in range(1,drug_num+1) :
        for n in range(0,m-1) :
            attr1.loc[attr_pos1,0] = int(m-1)
            attr1.loc[attr_pos1,1] = int(n)
            attr_pos1 = attr_pos1+1

    attr_pos2=0
    attr2=pd.DataFrame(np.zeros(shape=(int(drug_num*(drug_num-1)/2),1)))
    for i in range(1,int(drug_num*(drug_num-1)/2)+1) :
        attr2.loc[attr_pos2,0] =i
        attr_pos2 = attr_pos2+1        
    attr=pd.concat([attr2,attr1,attr],axis=1,ignore_index=True)
    attr=attr.rename(columns = {43:'label'}) 

    #print(attr)
    return attr

def getTL(DG_mat, DD_mat, GG_mat,DT_mat,TG_mat,TT_mat,i):

    WDD = DD_mat
    WDG = DG_mat
    WGD = DG_mat.T
    WGG = GG_mat
    WTT = TT_mat
    WTG = TG_mat
    WGT = TG_mat.T
    WDT = DT_mat
    WTD = DT_mat.T
    
    YD = WDD
    YG = WGD
    YT = WTD 

    DD = np.sum(WDD, axis=1)
    GG = np.sum(WGG, axis=1)
    TT = np.sum(WTT, axis=1)
    DG = np.sum(WDG, axis=1)
    GD = np.sum(WGD, axis=1)
    DT = np.sum(WDT, axis=1)
    TD = np.sum(WTD, axis=1)
    GT = np.sum(WGT, axis=1)
    TG = np.sum(WTG, axis=1)

    DD = np.diagflat(DD)
    GG = np.diagflat(GG)
    TT = np.diagflat(TT)
    DG = np.diagflat(DG)
    GD = np.diagflat(GD)
    DT = np.diagflat(DT)
    TD = np.diagflat(TD)
    GT = np.diagflat(GT)
    TG = np.diagflat(TG)

    D_DD = fix_no_zero(DD)
    D_GG = fix_no_zero(GG)
    D_TT = fix_no_zero(TT)
    D_DG = fix_no_zero(DG)
    D_GD = fix_no_zero(GD)
    D_DT = fix_no_zero(DT)
    D_TD = fix_no_zero(TD)
    D_GT = fix_no_zero(GT)
    D_TG = fix_no_zero(TG)

    SDD = np.dot(np.dot(D_DD,WDD),D_DD)
    SGG = np.dot(np.dot(D_GG,WGG),D_GG)
    STT = np.dot(np.dot(D_TT,WTT),D_TT)
    SDG = np.dot(np.dot(D_DG,WDG),D_GD)
    SGD = np.dot(np.dot(D_GD,WGD),D_DG)
    STD = np.dot(np.dot(D_TD,WTD),D_DT)
    SDT = np.dot(np.dot(D_DT,WDT),D_TD)
    SGT = np.dot(np.dot(D_GT,WGT),D_TG)
    STG = np.dot(np.dot(D_TG,WTG),D_GT)
   
    FDt1 = YD
    FGt1 = YG
    FTt1 = YT
    ID = np.eye(YD.shape[0])
    IG = np.eye(YG.shape[0])
    IT = np.eye(YT.shape[0])
    
    lamdaDD = 1
    lamdaGG = 1
    lamdaTT = 1
    lamdaTG = 0.01
    lamdaGT = lamdaTG    
    lamdaDG = 0.01
    lamdaGD = lamdaDG   
    lamdaDT = 0.01
    lamdaTD = lamdaDT

    it=1
    while it > 0.01:

        FDt2 = np.linalg.inv((lamdaDT+lamdaDG+1)*ID+2*lamdaDD*(ID-SDD))@(lamdaDG*SDG@FGt1+lamdaDT*SDT@FCt1+YD)
        FGt2 = np.linalg.inv((lamdaGT+lamdaGD+1)*IG+2*lamdaGG*(IG-SGG))@(lamdaGD*SGD@FDt1+lamdaGT*SGT@FCt1+YG)
        FTt2 = np.linalg.inv((lamdaTD+lamdaTG+1)*IT+2**lamdaTT*(IT-STT))@(lamdaTG*STG@FGt1+lamdaTD*STD@FDt1+YT)
        it = sqrt(np.sum(np.square(FDt2-FDt1)))
        FDt1 = FDt2
        FGt1 = FGt2
        FCt1 = FTt2
        #print(it)
    
    FDD =FDt2 

    return FDD