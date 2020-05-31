#!/usr/bin/env python
# coding: utf-8

# # Association Rule Mining - Raktim Mukhopadhyay - ASU ID- 1217167380

# In[1]:


import csv
import pandas as pd
from datetime import *
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import re


# In[2]:


def convertHumanReadable(date):
    humanReadableDate = (datetime.fromordinal(int(date)) + timedelta(days=date%1) - timedelta(days = 366)).strftime("%Y-%m-%d %H:%M:%S")
    return humanReadableDate


# In[3]:


def dataframeDateHumanReadable(df):
    for i in df.columns:
        df[i] = df[i].apply(lambda x: convertHumanReadable(x) if pd.notnull(x) else x)


# In[4]:


def secs_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
    return abs((d2 - d1).total_seconds())


# In[5]:


def correct_rule_sequence(string):
    strings_list = string.split(',')
    if ('CGM_0' in strings_list[0]):
        temp = strings_list[0]
        strings_list[0] = strings_list[1]
        strings_list[1] = temp
    else:
        pass
    return strings_list


# In[6]:


def remove_identifier(string):
    string = str(string)
    string = string.replace('CGM_M-','')
    string = string.replace('CGM_0-','')
    string = string.replace('I_B-','')
    string = string.replace('[','')
    string = string.replace(']','')
    string = string.replace("'",'')
    return string


# In[7]:


def extract_sixth_val(CGMSeriesLunchPat):
    sixth_val=[]
    index_list =[]
    for i in range(CGMSeriesLunchPat.shape[0]):
        if (len(CGMSeriesLunchPat.iloc[i].dropna())>=20):
            sixth_val.append(CGMSeriesLunchPat.iloc[i].dropna()[::-1][5])
            index_list.append(i)
    return sixth_val , index_list


# In[8]:


def max_cgm_level_data(CGMSeriesLunchPat):
    maxCGMLevel = []
    for i in range(CGMSeriesLunchPat.shape[0]):
        maxCGMLevel.append(max(CGMSeriesLunchPat.iloc[i].dropna()))
    return maxCGMLevel


# In[9]:


def non_zero_bolus_indices(InsulinBolusLunchPat):
    complete_non_zero =[]
    for i in range(InsulinBolusLunchPat.shape[0]):
        not_zero = []
        for j in range(len(InsulinBolusLunchPat.iloc[i].dropna())):
            if(InsulinBolusLunchPat.iloc[i].dropna()[j]!=0):
                not_zero.append(j)
        complete_non_zero.append(not_zero)
    return complete_non_zero


# In[10]:


def IB_values(InsulinDatenumLunchPat,non_zero_bolus_pat,InsulinBolusLunchPat,sixth_date_val_pat):
    ib_values=[]
    for i in range(len(non_zero_bolus_pat)):
        if (len(non_zero_bolus_pat[i])==1):
            max_val = max(InsulinBolusLunchPat.iloc[i])
            ib_values.append(max_val)
        elif (len(non_zero_bolus_pat[i])>1):
            intermediate_list=[]
            interest_points = InsulinDatenumLunchPat.iloc[i][non_zero_bolus_pat[i]]
            for k in interest_points:
                intermediate_list.append(secs_between(k,sixth_date_val_pat[i]))
            minpos = intermediate_list.index(min(intermediate_list))
            m = non_zero_bolus_pat[i][minpos]
            ib_values.append(InsulinBolusLunchPat.iloc[i][m])
    return ib_values


# In[11]:


def extract_sixth_val_date(CGMDatenumLunchPat,index_list):
    sixth_date_val=[]
    for i in index_list:
        sixth_date_val.append(CGMDatenumLunchPat.iloc[i].dropna()[::-1][5])
    return sixth_date_val


# In[12]:


def itemset_formatting(string):
    string = str(string).replace('frozenset','')
    string = string.replace("({",'')
    string = string.replace("})",'')
    string = string.replace("'",'')
    string_list = string.split(',')
    
    if(len(string_list)==3):
        cgm_m = 0
        cgm_o = 0
        ib = 0
        for i in range(len(string_list)):
            if ('CGM' in string_list[i]):
                if('CGM_M' in string_list[i]):
                    cgm_m = string_list[i].replace("CGM_M-",'')
                else:
                    cgm_o = string_list[i].replace("CGM_0-",'')
            elif ('I_B' in string_list[i]):
                ib = string_list[i].replace("I_B-",'')
        formatted = "{" + str(cgm_m)+","+str(cgm_o)+","+str(ib) + "}"
    
    elif(len(string_list)==2):
        cgm= 0
        ib = 0
        for i in range(len(string_list)):
            if ('CGM' in string_list[i]):
                if('CGM_M' in string_list[i]):
                    cgm = string_list[i].replace("CGM_M-",'')
                else:
                    cgm = string_list[i].replace("CGM_0-",'')
            elif ('I_B' in string_list[i]):
                ib = string_list[i].replace("I_B-",'')
        formatted = "{" + str(cgm)+","+str(ib) + "}"
        
    elif(len(string_list)==1):
        val= 0
        for i in range(len(string_list)):
            if ('CGM' in string_list[i]):
                if('CGM_M' in string_list[i]):
                    val = string_list[i].replace("CGM_M-",'')
                else:
                    val = string_list[i].replace("CGM_0-",'')
            elif ('I_B' in string_list[i]):
                val = string_list[i].replace("I_B-",'')
        formatted = "{" + str(val) + "}"
    return formatted


# In[13]:


CGMSeriesLunchPat1 = pd.read_csv('CGMSeriesLunchPat1.csv')
CGMSeriesLunchPat2 = pd.read_csv('CGMSeriesLunchPat2.csv')
CGMSeriesLunchPat3 = pd.read_csv('CGMSeriesLunchPat3.csv')
CGMSeriesLunchPat4 = pd.read_csv('CGMSeriesLunchPat4.csv')
CGMSeriesLunchPat5 = pd.read_csv('CGMSeriesLunchPat5.csv')


# In[14]:


CGMDatenumLunchPat1= pd.read_csv('CGMDatenumLunchPat1.csv')
CGMDatenumLunchPat2= pd.read_csv('CGMDatenumLunchPat2.csv')
CGMDatenumLunchPat3= pd.read_csv('CGMDatenumLunchPat3.csv')
CGMDatenumLunchPat4= pd.read_csv('CGMDatenumLunchPat4.csv')
CGMDatenumLunchPat5= pd.read_csv('CGMDatenumLunchPat5.csv')


# In[15]:


dataframeDateHumanReadable(CGMDatenumLunchPat1)
dataframeDateHumanReadable(CGMDatenumLunchPat2)
dataframeDateHumanReadable(CGMDatenumLunchPat3)
dataframeDateHumanReadable(CGMDatenumLunchPat4)
dataframeDateHumanReadable(CGMDatenumLunchPat5)


# In[16]:


InsulinBolusLunchPat1 = pd.read_csv('InsulinBolusLunchPat1.csv')
InsulinBolusLunchPat2 = pd.read_csv('InsulinBolusLunchPat2.csv')
InsulinBolusLunchPat3 = pd.read_csv('InsulinBolusLunchPat3.csv')
InsulinBolusLunchPat4 = pd.read_csv('InsulinBolusLunchPat4.csv')
InsulinBolusLunchPat5 = pd.read_csv('InsulinBolusLunchPat5.csv')


# In[17]:


InsulinDatenumLunchPat1= pd.read_csv('InsulinDatenumLunchPat1.csv')
InsulinDatenumLunchPat2= pd.read_csv('InsulinDatenumLunchPat2.csv')
InsulinDatenumLunchPat3= pd.read_csv('InsulinDatenumLunchPat3.csv')
InsulinDatenumLunchPat4= pd.read_csv('InsulinDatenumLunchPat4.csv')
InsulinDatenumLunchPat5= pd.read_csv('InsulinDatenumLunchPat5.csv')


# In[18]:


dataframeDateHumanReadable(InsulinDatenumLunchPat1)
dataframeDateHumanReadable(InsulinDatenumLunchPat2)
dataframeDateHumanReadable(InsulinDatenumLunchPat3)
dataframeDateHumanReadable(InsulinDatenumLunchPat4)
dataframeDateHumanReadable(InsulinDatenumLunchPat5)


# In[19]:


sixth_val_pat1,indices_pat1 = extract_sixth_val(CGMSeriesLunchPat1)
sixth_val_pat2,indices_pat2 = extract_sixth_val(CGMSeriesLunchPat2)
sixth_val_pat3,indices_pat3 = extract_sixth_val(CGMSeriesLunchPat3)
sixth_val_pat4,indices_pat4 = extract_sixth_val(CGMSeriesLunchPat4)
sixth_val_pat5,indices_pat5 = extract_sixth_val(CGMSeriesLunchPat5)


# In[20]:


CGMSeriesLunchPat1 = CGMSeriesLunchPat1.iloc[indices_pat1].reset_index().drop(columns=['index'])
CGMSeriesLunchPat2 = CGMSeriesLunchPat2.iloc[indices_pat2].reset_index().drop(columns=['index'])
CGMSeriesLunchPat3 = CGMSeriesLunchPat3.iloc[indices_pat3].reset_index().drop(columns=['index'])
CGMSeriesLunchPat4 = CGMSeriesLunchPat4.iloc[indices_pat4].reset_index().drop(columns=['index'])
CGMSeriesLunchPat5 = CGMSeriesLunchPat5.iloc[indices_pat5].reset_index().drop(columns=['index'])


# In[21]:


maxCGMLevelpat1 = pd.DataFrame(max_cgm_level_data(CGMSeriesLunchPat1)).rename(columns={0:'CGM_M'}).astype(float)
maxCGMLevelpat2 = pd.DataFrame(max_cgm_level_data(CGMSeriesLunchPat2)).rename(columns={0:'CGM_M'}).astype(float)
maxCGMLevelpat3 = pd.DataFrame(max_cgm_level_data(CGMSeriesLunchPat3)).rename(columns={0:'CGM_M'}).astype(float)
maxCGMLevelpat4 = pd.DataFrame(max_cgm_level_data(CGMSeriesLunchPat4)).rename(columns={0:'CGM_M'}).astype(float)
maxCGMLevelpat5 = pd.DataFrame(max_cgm_level_data(CGMSeriesLunchPat5)).rename(columns={0:'CGM_M'}).astype(float)


# In[22]:


sixth_val_pat1 = pd.DataFrame(sixth_val_pat1).rename(columns={0:'CGM_0'}).astype(float)
sixth_val_pat2 = pd.DataFrame(sixth_val_pat2).rename(columns={0:'CGM_0'}).astype(float)
sixth_val_pat3 = pd.DataFrame(sixth_val_pat3).rename(columns={0:'CGM_0'}).astype(float)
sixth_val_pat4 = pd.DataFrame(sixth_val_pat4).rename(columns={0:'CGM_0'}).astype(float)
sixth_val_pat5 = pd.DataFrame(sixth_val_pat5).rename(columns={0:'CGM_0'}).astype(float)


# In[23]:


sixth_date_val_pat1= extract_sixth_val_date(CGMDatenumLunchPat1,indices_pat1)
sixth_date_val_pat2= extract_sixth_val_date(CGMDatenumLunchPat2,indices_pat2)
sixth_date_val_pat3= extract_sixth_val_date(CGMDatenumLunchPat3,indices_pat3)
sixth_date_val_pat4= extract_sixth_val_date(CGMDatenumLunchPat4,indices_pat4)
sixth_date_val_pat5= extract_sixth_val_date(CGMDatenumLunchPat5,indices_pat5)


# In[24]:


InsulinDatenumLunchPat1 = InsulinDatenumLunchPat1.iloc[indices_pat1].reset_index().drop(columns=['index'])
InsulinDatenumLunchPat2 = InsulinDatenumLunchPat2.iloc[indices_pat2].reset_index().drop(columns=['index'])
InsulinDatenumLunchPat3 = InsulinDatenumLunchPat3.iloc[indices_pat3].reset_index().drop(columns=['index'])
InsulinDatenumLunchPat4 = InsulinDatenumLunchPat4.iloc[indices_pat4].reset_index().drop(columns=['index'])
InsulinDatenumLunchPat5 = InsulinDatenumLunchPat5.iloc[indices_pat5].reset_index().drop(columns=['index'])


# In[25]:


InsulinBolusLunchPat1 = InsulinBolusLunchPat1.iloc[indices_pat1].reset_index().drop(columns=['index'])
InsulinBolusLunchPat2 = InsulinBolusLunchPat2.iloc[indices_pat2].reset_index().drop(columns=['index'])
InsulinBolusLunchPat3 = InsulinBolusLunchPat3.iloc[indices_pat3].reset_index().drop(columns=['index'])
InsulinBolusLunchPat4 = InsulinBolusLunchPat4.iloc[indices_pat4].reset_index().drop(columns=['index'])
InsulinBolusLunchPat5 = InsulinBolusLunchPat5.iloc[indices_pat5].reset_index().drop(columns=['index'])


# In[26]:


non_zero_bolus_pat1 = non_zero_bolus_indices(InsulinBolusLunchPat1)
non_zero_bolus_pat2 = non_zero_bolus_indices(InsulinBolusLunchPat2)
non_zero_bolus_pat3 = non_zero_bolus_indices(InsulinBolusLunchPat3)
non_zero_bolus_pat4 = non_zero_bolus_indices(InsulinBolusLunchPat4)
non_zero_bolus_pat5 = non_zero_bolus_indices(InsulinBolusLunchPat5)


# In[27]:


IB_values_pat1 = pd.DataFrame(IB_values(InsulinDatenumLunchPat1,non_zero_bolus_pat1,InsulinBolusLunchPat1,sixth_date_val_pat1)).rename(columns={0:'I_B'})
IB_values_pat2 = pd.DataFrame(IB_values(InsulinDatenumLunchPat2,non_zero_bolus_pat2,InsulinBolusLunchPat2,sixth_date_val_pat2)).rename(columns={0:'I_B'})
IB_values_pat3 = pd.DataFrame(IB_values(InsulinDatenumLunchPat3,non_zero_bolus_pat3,InsulinBolusLunchPat3,sixth_date_val_pat3)).rename(columns={0:'I_B'})
IB_values_pat4 = pd.DataFrame(IB_values(InsulinDatenumLunchPat4,non_zero_bolus_pat4,InsulinBolusLunchPat4,sixth_date_val_pat4)).rename(columns={0:'I_B'})
IB_values_pat5 = pd.DataFrame(IB_values(InsulinDatenumLunchPat5,non_zero_bolus_pat5,InsulinBolusLunchPat5,sixth_date_val_pat5)).rename(columns={0:'I_B'})


# In[28]:


dataset_pat1 = pd.concat([maxCGMLevelpat1,sixth_val_pat1,IB_values_pat1],axis=1)
dataset_pat2 = pd.concat([maxCGMLevelpat2,sixth_val_pat2,IB_values_pat2],axis=1)
dataset_pat3 = pd.concat([maxCGMLevelpat3,sixth_val_pat3,IB_values_pat3],axis=1)
dataset_pat4 = pd.concat([maxCGMLevelpat4,sixth_val_pat4,IB_values_pat4],axis=1)
dataset_pat5 = pd.concat([maxCGMLevelpat5,sixth_val_pat5,IB_values_pat5],axis=1)


# In[29]:


dataset_pat1.head()


# In[30]:


def create_bin(x):
    if 50<x<=60:
        x=1
    elif 60<x<=70:
        x=2
    elif 70<x<=80:
        x=3
    elif 80<x<=90:
        x=4
    elif 90<x<=100:
        x=5
    elif 100<x<=110:
        x=6
    elif 110<x<=120:
        x=7
    elif 120<x<=130:
        x=8
    elif 130<x<=140:
        x=9
    elif 140<x<=150:
        x=10
    elif 150<x<=160:
        x=11
    elif 160<x<=170:
        x=12
    elif 170<x<=180:
        x=13
    elif 180<x<=190:
        x=14
    elif 190<x<=200:
        x=15
    elif 200<x<=210:
        x=16
    elif 210<x<=220:
        x=17
    elif 220<x<=230:
        x=18
    elif 230<x<=240:
        x=19
    elif 240<x<=250:
        x=20
    elif 250<x<=260:
        x=21
    elif 260<x<=270:
        x=22
    elif 270<x<=280:
        x =23
    elif 280<x<=290:
        x=24
    elif 290<x<=300:
        x=25
    elif 300<x<=310:
        x=26
    elif 310<x<=320:
        x=27
    elif 320<x<=330:
        x=28
    elif 330<x<=340:
        x=29
    elif 340<x<=350:
        x=30
    elif 350<x<=360:
        x=31
    elif 360<x<=370:
        x=32
    elif 370<x<=380:
        x=33
    elif 380<x<=390:
        x=34
    elif 390<x<=400:
        x=35
    return x


# In[31]:


dataset_pat1['CGM_M'] = dataset_pat1['CGM_M'].apply(create_bin)
dataset_pat1['CGM_0'] = dataset_pat1['CGM_0'].apply(create_bin)


# In[32]:


dataset_pat2['CGM_M'] = dataset_pat2['CGM_M'].apply(create_bin)
dataset_pat2['CGM_0'] = dataset_pat2['CGM_0'].apply(create_bin)


# In[33]:


dataset_pat3['CGM_M'] = dataset_pat3['CGM_M'].apply(create_bin)
dataset_pat3['CGM_0'] = dataset_pat3['CGM_0'].apply(create_bin)


# In[34]:


dataset_pat4['CGM_M'] = dataset_pat4['CGM_M'].apply(create_bin)
dataset_pat4['CGM_0'] = dataset_pat4['CGM_0'].apply(create_bin)


# In[35]:


dataset_pat5['CGM_M'] = dataset_pat5['CGM_M'].apply(create_bin)
dataset_pat5['CGM_0'] = dataset_pat5['CGM_0'].apply(create_bin)


# In[36]:


def add_identifier(dataset):
    dataset['CGM_M'] = dataset['CGM_M'].astype(str)
    dataset['CGM_0'] = dataset['CGM_0'].astype(str)
    for i in range(dataset.shape[0]):
        dataset['CGM_M'].iloc[i]= 'CGM_M-'+ str(dataset['CGM_M'].iloc[i])
        dataset['CGM_0'].iloc[i]= 'CGM_0-'+ str(dataset['CGM_0'].iloc[i])
        dataset['I_B'].iloc[i]= 'I_B-' + str(dataset['I_B'].iloc[i])
    return dataset


# In[37]:


def rule_mining(dataset,min_sup):
    records = []
    for i in range(dataset.shape[0]):
        records.append([str(dataset.values[i,j]) for j in range(0,3)])
    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support= min_sup, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold = min_sup)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules = rules[(rules['antecedent_len'] >= 2)]
    rules['antecedents'] = rules['antecedents'].astype(str)
    rules['consequents'] = rules['consequents'].astype(str)
    rules = rules[~rules["antecedents"].str.contains("I_B")]
    rules = rules[~rules["consequents"].str.contains("CGM_M")]
    rules = rules[~rules["consequents"].str.contains("CGM_0")]
    for i in range(rules.shape[0]):
        rules['antecedents'].iloc[i]=rules['antecedents'].iloc[i].replace('frozenset','')
        rules['consequents'].iloc[i]=rules['consequents'].iloc[i].replace('frozenset','')
        rules['antecedents'].iloc[i]=rules['antecedents'].iloc[i].replace('({','')
        rules['antecedents'].iloc[i]=rules['antecedents'].iloc[i].replace('})','')
        rules['consequents'].iloc[i]=rules['consequents'].iloc[i].replace('({','')
        rules['consequents'].iloc[i]=rules['consequents'].iloc[i].replace('})','')
        rules['antecedents'].iloc[i]=rules['antecedents'].iloc[i].replace("'",'')
        rules['consequents'].iloc[i]=rules['consequents'].iloc[i].replace("'",'')
    rules = rules.reset_index().drop(columns=['index'])
    rules['antecedents'] = rules ['antecedents'].apply(correct_rule_sequence)
    rules['antecedents'] = rules ['antecedents'].apply(remove_identifier)
    rules['consequents'] = rules ['consequents'].apply(remove_identifier)
    rules['formatted_rules'] = "{"+rules['antecedents']+" -> "+rules['consequents']+"}"
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(itemset_formatting)
    return rules,frequent_itemsets


# In[38]:


dataset_pat1 = add_identifier(dataset_pat1)
dataset_pat2 = add_identifier(dataset_pat2)
dataset_pat3 = add_identifier(dataset_pat3)
dataset_pat4 = add_identifier(dataset_pat4)
dataset_pat5 = add_identifier(dataset_pat5)


# In[39]:


rules_pat1,f_tems_pat1=rule_mining(dataset_pat1,0.001)
rules_pat2,f_tems_pat2=rule_mining(dataset_pat2,0.001)
rules_pat3,f_tems_pat3=rule_mining(dataset_pat3,0.001)
rules_pat4,f_tems_pat4=rule_mining(dataset_pat4,0.001)
rules_pat5,f_tems_pat5=rule_mining(dataset_pat5,0.001)


# In[40]:


# Frequent Itemsets
f_tems_pat1.head()


# In[41]:


f_tems_pat2.head()


# In[42]:


f_tems_pat3.head()


# In[43]:


f_tems_pat4.head()


# In[44]:


f_tems_pat5.head()


# In[45]:


#Association Rules
rules_pat1.head()


# In[46]:


rules_pat2.head()


# In[47]:


rules_pat3.head()


# In[48]:


rules_pat4.head()


# In[49]:


rules_pat5.head()


# In[50]:


#Rules ranked by confidence
rules_pat1.sort_values(by=['confidence'],ascending=False).head()


# In[51]:


rules_pat2.sort_values(by=['confidence'],ascending=False).head()


# In[52]:


rules_pat3.sort_values(by=['confidence'],ascending=False).head()


# In[53]:


rules_pat4.sort_values(by=['confidence'],ascending=False).head()


# In[54]:


rules_pat5.sort_values(by=['confidence'],ascending=False).head()


# In[55]:


#Anomalous Events
rules_pat1[rules_pat1['confidence']==min(rules_pat1['confidence'])].head()


# In[56]:


rules_pat2[rules_pat2['confidence']==min(rules_pat2['confidence'])].head()


# In[57]:


rules_pat3[rules_pat3['confidence']==min(rules_pat3['confidence'])].head()


# In[58]:


rules_pat4[rules_pat4['confidence']==min(rules_pat4['confidence'])].head()


# In[59]:


rules_pat5[rules_pat5['confidence']==min(rules_pat5['confidence'])].head()


# # Submission CSVs

# In[66]:


#Report the most frequent itemsets for each of the subjects (Bin for CGM_M, Bin for CGM_0, Insulin Bolus)


# In[67]:


frequent_itemsets = pd.concat([f_tems_pat1,f_tems_pat2,f_tems_pat3,f_tems_pat4,f_tems_pat5]).reset_index().drop(columns=['index'])


# In[68]:


frequent_itemsets = frequent_itemsets.drop_duplicates()[['itemsets']]


# In[80]:


frequent_itemsets.to_csv('frequent_itemsets.csv',index=False,header=None)


# In[69]:


# CSV file with largest confidence rules. One row for each rule. Rules are of the form {Bin for CGM_M,Bin for CGM_0 }→I_B


# In[70]:


larg_conf_rules_pat1 = rules_pat1[rules_pat1['confidence']==max(rules_pat1['confidence'])]
larg_conf_rules_pat2 = rules_pat2[rules_pat2['confidence']==max(rules_pat2['confidence'])]
larg_conf_rules_pat3 = rules_pat3[rules_pat3['confidence']==max(rules_pat3['confidence'])]
larg_conf_rules_pat4 = rules_pat4[rules_pat4['confidence']==max(rules_pat4['confidence'])]
larg_conf_rules_pat5 = rules_pat5[rules_pat5['confidence']==max(rules_pat5['confidence'])]


# In[71]:


larg_conf_rules = pd.concat([larg_conf_rules_pat1,larg_conf_rules_pat2,larg_conf_rules_pat3,larg_conf_rules_pat4,larg_conf_rules_pat5]).reset_index().drop(columns=['index'])


# In[72]:


larg_conf_rules = larg_conf_rules[['formatted_rules']].drop_duplicates()


# In[81]:


larg_conf_rules.to_csv('largest_confidence_rules.csv',index=False,header=None)


# In[74]:


min_conf_rules_pat1 = rules_pat1[rules_pat1['confidence']==min(rules_pat1['confidence'])]
min_conf_rules_pat2 = rules_pat2[rules_pat2['confidence']==min(rules_pat2['confidence'])]
min_conf_rules_pat3 = rules_pat3[rules_pat3['confidence']==min(rules_pat3['confidence'])]
min_conf_rules_pat4 = rules_pat4[rules_pat4['confidence']==min(rules_pat4['confidence'])]
min_conf_rules_pat5 = rules_pat5[rules_pat5['confidence']==min(rules_pat5['confidence'])]


# In[75]:


min_conf_rules = pd.concat([min_conf_rules_pat1,min_conf_rules_pat2,min_conf_rules_pat3,min_conf_rules_pat4,min_conf_rules_pat5])[['formatted_rules']].reset_index().drop(columns=['index'])


# In[83]:


min_conf_rules.to_csv('anomalous_rules.csv',index=False,header=None)

