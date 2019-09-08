#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:18:32 2019

@author: tushar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:32:05 2019

@author: tushar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import re
import string
import math

def gaussian(marks, mean, std):
    temp = (marks - mean)**2
    temp = -temp
    ex_den = 2 * (std**2)
    ex = np.exp(temp/ex_den)
    den = np.sqrt(2 * math.pi)
    den = den * std
    den = 1 / den;
    prob = den * ex
    
    return prob
    

def standard_deviation(X):
    m = np.mean(X)
    n = len(X)
    '''for i in range(n):
        temp = np.sum((X[i] - m)**2)
    temp = temp / (n-1)
    std = np.sqrt(temp)'''
    temp = np.sum(pow(X-m, 2))
    temp = temp / n
    std = np.sqrt(temp)
    
    return std

def prediction(X_test):
    result = []
    r = len(X_test)
    #print(r)

    for i in range(0, r):
        gen = X_test.iloc[i, 1]
        if (gen == 1):
            prob_gen_conti = math.log(prob_M_g_conti)
        else:
            prob_gen_conti = math.log(prob_F_g_conti)
        
        cast = X_test.iloc[i, 2]
        if (cast == 'SC'):
            prob_cast_conti = math.log(prob_SC_g_conti)
        elif (cast == 'ST'):
            prob_cast_conti = math.log(prob_ST_g_conti)
        elif (cast == 'OC'):
            prob_cast_conti = math.log(prob_OC_g_conti)
        elif (cast == 'BC'):
            prob_cast_conti = math.log(prob_BC_g_conti)
            
        math_marks = X_test.iloc[i, 3]
        prob_math_conti = math.log(gaussian(math_marks, mean_math_conti, math_conti_std))
        #prob_math_conti = math.log(prob_math_conti)
        
        eng_marks = X_test.iloc[i, 4]
        prob_eng_conti = math.log(gaussian(eng_marks, mean_eng_conti, eng_conti_std))
        
        science_marks = X_test.iloc[i, 5]
        prob_science_conti = math.log(gaussian(science_marks, mean_science_conti, science_conti_std))
    
        net = X_test.iloc[i, 6]
        if (net == True):
            prob_net_conti = math.log(prob_internet_conti)
        else:
            prob_net_conti = math.log(prob_notinternet_conti)
            
        guardian = X_test.iloc[i, 7]
        if (guardian == 0):
            prob_guardian_conti = math.log(prob_mother_conti)
        elif (guardian == 1):
            prob_guardian_conti = math.log(prob_father_conti)
        else:
            prob_guardian_conti = math.log(prob_other_conti)
            
        study = X_test.iloc[i, 8]
        if (study == 1):
            prob_study_conti = math.log(prob_study1_conti)
        elif (study == 2):
            prob_study_conti = math.log(prob_study2_conti)
        elif (study == 3):
            prob_study_conti = math.log(prob_study3_conti)
        else:
            prob_study_conti = math.log(prob_study4_conti)
            
        fails = X_test.iloc[i, 9]
        if (fails == 0):
            prob_fails_conti = math.log(prob_fail0_conti)
        elif (fails == 1):
            prob_fails_conti = math.log(prob_fail1_conti)
        elif (fails == 2):
            prob_fails_conti = math.log(prob_fail2_conti)
        else:
            prob_fails_conti = math.log(prob_fail3_conti)
                    
        schoolsup = X_test.iloc[i, 10]
        if (schoolsup == 'yes'):
            prob_school_conti = math.log(prob_schoolsup_conti)
        else:
            prob_school_conti = math.log(prob_schoolnotsup_conti)
            
        famsup = X_test.iloc[i, 11]
        if (famsup == 'yes'):
            prob_family_conti = math.log(prob_famsup_conti)
        else:
            prob_family_conti = math.log(prob_famnotsup_conti)
            
        absences = X_test.iloc[i, 12]
        prob_absence_conti = math.log(gaussian(absences, mean_absences_conti, absences_conti_std))

        freetime = X_test.iloc[i, 13]
        if (freetime == 1):
            prob_freetime_conti = math.log(prob_freetime1_conti)
        elif (freetime == 2):
            prob_freetime_conti = math.log(prob_freetime2_conti)
        elif (freetime == 3):
            prob_freetime_conti = math.log(prob_freetime3_conti)
        elif (freetime == 4):
            prob_freetime_conti = math.log(prob_freetime4_conti)
        else:
            prob_freetime_conti = math.log(prob_freetime5_conti)
            
        goout = X_test.iloc[i, 14]
        if (goout == 5):
            prob_goout_conti = math.log(prob_goout5_conti)
        elif (goout == 1):
            prob_goout_conti = math.log(prob_goout1_conti)
        elif (goout == 2):
            prob_goout_conti = math.log(prob_goout2_conti)
        elif (goout == 3):
            prob_goout_conti = math.log(prob_goout3_conti)
        else:
            prob_goout_conti = math.log(prob_goout4_conti)
            
        health = X_test.iloc[i, 15]
        if (health == 5):
            prob_health_conti = math.log(prob_health5_conti)
        elif (health == 1):
            prob_health_conti = math.log(prob_health1_conti)
        elif (health == 2):
            prob_health_conti = math.log(prob_health2_conti)
        elif (health == 3):
            prob_health_conti = math.log(prob_health3_conti)
        else:
            prob_health_conti = math.log(prob_health4_conti)
            
        walc = X_test.iloc[i, 16]
        if (walc == 5):
            prob_walc_conti = math.log(prob_walc5_conti)
        elif (walc == 1):
            prob_walc_conti = math.log(prob_walc1_conti)
        elif (walc == 2):
            prob_walc_conti = math.log(prob_walc2_conti)
        elif (walc == 3):
            prob_walc_conti = math.log(prob_walc3_conti)
        else:
            prob_walc_conti = math.log(prob_walc4_conti)
        
        
        prob_continue = (prob_gen_conti + prob_cast_conti + prob_math_conti + prob_eng_conti + prob_science_conti + prob_net_conti + prob_guardian_conti + prob_study_conti + prob_fails_conti + prob_school_conti + prob_family_conti + prob_absence_conti + prob_freetime_conti + prob_goout_conti + prob_health_conti + prob_walc_conti +prob['continue'])
        
        #for drop probability
        if (gen == '1'):
            prob_gen_drop = math.log(prob_M_g_drop)
        else:
            prob_gen_drop = math.log(prob_F_g_drop)
            
        if (cast == 'SC'):
            prob_cast_drop = math.log(prob_SC_g_drop)
        elif (cast == 'ST'):
            prob_cast_drop = math.log(prob_ST_g_drop)
        elif (cast == 'OC'):
            prob_cast_drop = math.log(prob_OC_g_drop)
        elif (cast == 'BC'):
            prob_cast_drop = math.log(prob_BC_g_drop)
            
        prob_math_drop = math.log(gaussian(math_marks, mean_math_drop, math_drop_std))
        
        prob_eng_drop = math.log(gaussian(eng_marks, mean_eng_drop, eng_drop_std))
        
        prob_science_drop = math.log(gaussian(science_marks, mean_science_drop, science_drop_std))
    
        if (net == True):
            prob_net_drop = math.log(prob_internet_drop)
        else:
            prob_net_drop = math.log(prob_notinternet_drop)
            
        if (guardian == 0):
            prob_guardian_drop = math.log(prob_mother_drop)
        elif (guardian == 1):
            prob_guardian_drop = math.log(prob_father_drop)
        else:
            prob_guardian_drop = math.log(prob_other_drop)
            
        
        if (study == 1):
            prob_study_drop = math.log(prob_study1_drop)
        elif (study == 2):
            prob_study_drop = math.log(prob_study2_drop)
        elif (study == 3):
            prob_study_drop = math.log(prob_study3_drop)
        else:
            prob_study_drop = math.log(prob_study4_drop)
            
        if (fails == 0):
            prob_fails_drop = math.log(prob_fail0_drop)
        elif (fails == 1):
            prob_fails_drop = math.log(prob_fail1_drop)
        elif (fails == 2):
            prob_fails_drop = math.log(prob_fail2_drop)
        else:
            prob_fails_drop = math.log(prob_fail3_drop)
        
            
        if (schoolsup == 'yes'):
            prob_school_drop = math.log(prob_schoolsup_drop)
        else:
            prob_school_drop = math.log(prob_schoolnotsup_drop)
        
                   
        if (famsup == 'yes'):
            prob_family_drop = math.log(prob_famsup_drop)
        else:
            prob_family_drop = math.log(prob_famnotsup_drop)
            
        #absences = X_test.iloc[i, 12]
        prob_absence_drop = math.log(gaussian(absences, mean_absences_drop, absences_drop_std))

        #freetime = X_test.iloc[i, 13]        
        if (freetime == 1):
            prob_freetime_drop = math.log(prob_freetime1_drop)
        elif (freetime == 2):
            prob_freetime_drop = math.log(prob_freetime2_drop)
        elif (freetime == 3):
            prob_freetime_drop = math.log(prob_freetime3_drop)
        elif (freetime == 4):
            prob_freetime_drop = math.log(prob_freetime4_drop)
        else:
            prob_freetime_drop = math.log(prob_freetime5_drop)
            
        #goout = X_test.iloc[i, 14]
        if (goout == 5):
            prob_goout_drop = math.log(prob_goout5_drop)
        elif (goout == 1):
            prob_goout_drop = math.log(prob_goout1_drop)
        elif (goout == 2):
            prob_goout_drop = math.log(prob_goout2_drop)
        elif (goout == 3):
            prob_goout_drop = math.log(prob_goout3_drop)
        else:
            prob_goout_drop = math.log(prob_goout4_drop)
            
        #health = X_test.iloc[i, 15]
        if (health == 5):
            prob_health_drop = math.log(prob_health5_drop)
        elif (health == 1):
            prob_health_drop = math.log(prob_health1_drop)
        elif (health == 2):
            prob_health_drop = math.log(prob_health2_drop)
        elif (health == 3):
            prob_health_drop = math.log(prob_health3_drop)
        else:
            prob_health_drop = math.log(prob_health4_drop)
            
        #walc = X_test.iloc[i, 16]
        if (walc == 5):
            prob_walc_drop = math.log(prob_walc5_drop)
        elif (walc == 1):
            prob_walc_drop = math.log(prob_walc1_drop)
        elif (walc == 2):
            prob_walc_drop = math.log(prob_walc2_drop)
        elif (walc == 3):
            prob_walc_drop = math.log(prob_walc3_drop)
        else:
            prob_walc_drop = math.log(prob_walc4_drop)
        
        prob_drop = (prob_gen_drop + prob_cast_drop + prob_math_drop + prob_eng_drop + prob_science_drop + prob_net_drop+ prob_guardian_drop + prob_study_drop + prob_fails_drop + prob_school_drop + prob_family_drop + prob_absence_drop + prob_freetime_drop + prob_goout_drop + prob_health_drop + prob_walc_drop + prob['drop'])
        
        r = r-1
        #print(r)
        if (prob_continue > prob_drop):
            result.append(1)
        else:
            result.append(0)
            
    return result
    
    
    

df = pd.read_csv("/home/tushar/Documents/pythonProjects/student dropout/drop_data.csv")

df['continue_drop']=df.continue_drop.map({'continue':1,'drop':0})

df['gender']=df.gender.map({'M':1,'F':0})

df['guardian']=df.guardian.map({'mother':0,'father':1, 'other':2})

#df['internet']=df.internet.map({'True':1,'False':0})

'''temp={'continue':1,'drop':0}
df.continu_drop=[temp[i] for i in df.continue_drop]

temp={'M':1,'F':0}
df.gender=[temp[i] for i in df.gender]

temp={'True':1,'False':0}
df.internet=[temp[i] for i in df.internet]'''

X = df.iloc[:, 1:]
Y = df.iloc[:,0:1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

'''plt.scatter([df.iloc[:,4:5].values],[df.iloc[:,5:6].values],c='b')
plt.xlabel('X1')
plt.ylabel('x2')
#plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()'''

prob = {}
num_samples = {}

num_samples['continue'] = Y_train.continue_drop.value_counts()[1]
num_samples['drop'] = Y_train.continue_drop.value_counts()[0]

n = len(X_train)
prob['continue'] = math.log((num_samples['continue'] / n))
prob['drop'] = math.log((num_samples['drop'] / n))

#gender_prob['F'] = {}
#gender_prob['M'] = {}

#continue_data = []
drop_data = pd.DataFrame()

continue_data = pd.DataFrame()
#df_ = df_.fillna(0)
#ab = X_train.iloc[0,0:]
#print(X_train.iloc[1, ])

#continue_data = continue_data.append(ab)
#continue_data = continue_data.append(ab)

for i in range(n):
    y = Y_train.iloc[i,0]
    x = X_train.iloc[i]
    
    if y == 1:
        continue_data = continue_data.append(x)
    else:
        drop_data = drop_data.append(x)
        
        
#print(len(drop_data))
#print(len(continue_data))

#find prob(caste/continue)

num_caste_conti = {}

num_caste_conti['SC'] = continue_data.caste.value_counts()['SC']
num_caste_conti['ST'] = continue_data.caste.value_counts()['ST']
num_caste_conti['OC'] = continue_data.caste.value_counts()['OC']
num_caste_conti['BC'] = continue_data.caste.value_counts()['BC']

#print(num_caste_conti['SC'] + num_caste_conti['ST'] + num_caste_conti['OC'] + num_caste_conti['BC'])

prob_SC_g_conti = (num_caste_conti['SC']+1) / (len(continue_data)+4)
prob_ST_g_conti = (num_caste_conti['ST']+1) / (len(continue_data)+4)
prob_OC_g_conti = (num_caste_conti['OC']+1) / (len(continue_data)+4)
prob_BC_g_conti = (num_caste_conti['BC']+1) / (len(continue_data)+4)

#print(prob_SC_g_conti + prob_ST_g_conti + prob_OC_g_conti + prob_BC_g_conti)

#find prob(gender/continue)

num_gender_conti = {}

num_gender_conti['M'] = continue_data.gender.value_counts()[1]
num_gender_conti['F'] = continue_data.gender.value_counts()[0]

prob_M_g_conti = (num_gender_conti['M']+1) / (len(continue_data)+2)
prob_F_g_conti = (num_gender_conti['F']+1) / (len(continue_data)+2)

#print(prob_M_g_conti + prob_F_g_conti)

#find prob(internet/continue)

num_internet_conti = {}

num_internet_conti['yes'] = continue_data.internet.value_counts()[1]
num_internet_conti['no'] = continue_data.internet.value_counts()[0]

prob_internet_conti = (num_internet_conti['yes']+1) / (len(continue_data)+2)
prob_notinternet_conti = (num_internet_conti['no']+1) / (len(continue_data)+2)

#find standard deviation for english / continue

english_conti = continue_data['english_marks'].values

mean_eng_conti = np.mean(english_conti)
eng_conti_std = standard_deviation(english_conti)


#find standard deviation for mathematics / continue

mathe_conti = continue_data['mathematics_marks'].values

mean_math_conti = np.mean(mathe_conti)
math_conti_std = standard_deviation(mathe_conti)


#find standard deviation for science / continue

science_conti = continue_data['science_marks'].values

mean_science_conti = np.mean(science_conti)
science_conti_std = standard_deviation(science_conti)

#find prob(guardian/continue)

num_guardian_conti = {}

num_guardian_conti['0'] = continue_data.guardian.value_counts()[0]
num_guardian_conti['1'] = continue_data.guardian.value_counts()[1]
num_guardian_conti['2'] = continue_data.guardian.value_counts()[2]

prob_mother_conti = (num_guardian_conti['0']+1) / (len(continue_data)+3)
prob_father_conti = (num_guardian_conti['1']+1) / (len(continue_data)+3)
prob_other_conti = (num_guardian_conti['2']+1) / (len(continue_data)+3)

#find prob(study/continue)

num_study_conti = {}

#num_study_conti['0'] = continue_data.studytime.value_counts()[0]
num_study_conti['1'] = continue_data.studytime.value_counts()[1]
num_study_conti['2'] = continue_data.studytime.value_counts()[2]
num_study_conti['3'] = continue_data.studytime.value_counts()[3]
num_study_conti['4'] = continue_data.studytime.value_counts()[4]


#prob_study0_conti = (num_study_conti['0']+1) / (len(continue_data)+5)
prob_study1_conti = (num_study_conti['1']+1) / (len(continue_data)+5)
prob_study2_conti = (num_study_conti['2']+1) / (len(continue_data)+5)
prob_study3_conti = (num_study_conti['3']+1) / (len(continue_data)+5)
prob_study4_conti = (num_study_conti['4']+1) / (len(continue_data)+5)

#find prob(failures/continue)

num_failures_conti = {}

num_failures_conti['0'] = continue_data.failures.value_counts()[0]
num_failures_conti['1'] = continue_data.failures.value_counts()[1]
num_failures_conti['2'] = continue_data.failures.value_counts()[2]
num_failures_conti['3'] = continue_data.failures.value_counts()[3]
#num_failures_conti['4'] = continue_data.failures.value_counts()[4]


prob_fail0_conti = (num_failures_conti['0']+1) / (len(continue_data)+5)
prob_fail1_conti = (num_failures_conti['1']+1) / (len(continue_data)+5)
prob_fail2_conti = (num_failures_conti['2']+1) / (len(continue_data)+5)
prob_fail3_conti = (num_failures_conti['3']+1) / (len(continue_data)+5)
#prob_fail4_conti = (num_failures_conti['4']+1) / (len(continue_data)+5)

#find prob(schoolsup/continue)

num_schoolsup_conti = {}

num_schoolsup_conti['yes'] = continue_data.schoolsup.value_counts()['yes']
num_schoolsup_conti['no'] = continue_data.schoolsup.value_counts()['no']

prob_schoolsup_conti = (num_schoolsup_conti['yes']+1) / (len(continue_data)+2)
prob_schoolnotsup_conti = (num_schoolsup_conti['no']+1) / (len(continue_data)+2)

#find prob(famsup/continue)

num_famsup_conti = {}

num_famsup_conti['yes'] = continue_data.famsup.value_counts()['yes']
num_famsup_conti['no'] = continue_data.famsup.value_counts()['no']

prob_famsup_conti = (num_famsup_conti['yes']+1) / (len(continue_data)+2)
prob_famnotsup_conti = (num_famsup_conti['no']+1) / (len(continue_data)+2)

#find standard deviation for absences / continue

absences_conti = continue_data['absences'].values

mean_absences_conti = np.mean(absences_conti)
absences_conti_std = standard_deviation(absences_conti)

#find prob(freetime/continue)

num_freetime_conti = {}

num_freetime_conti['5'] = continue_data.freetime.value_counts()[5]
num_freetime_conti['1'] = continue_data.freetime.value_counts()[1]
num_freetime_conti['2'] = continue_data.freetime.value_counts()[2]
num_freetime_conti['3'] = continue_data.freetime.value_counts()[3]
num_freetime_conti['4'] = continue_data.freetime.value_counts()[4]


prob_freetime5_conti = (num_freetime_conti['5']+1) / (len(continue_data)+5)
prob_freetime1_conti = (num_freetime_conti['1']+1) / (len(continue_data)+5)
prob_freetime2_conti = (num_freetime_conti['2']+1) / (len(continue_data)+5)
prob_freetime3_conti = (num_freetime_conti['3']+1) / (len(continue_data)+5)
prob_freetime4_conti = (num_freetime_conti['4']+1) / (len(continue_data)+5)

#find prob(goout/continue)

num_goout_conti = {}

num_goout_conti['5'] = continue_data.goout.value_counts()[5]
num_goout_conti['1'] = continue_data.goout.value_counts()[1]
num_goout_conti['2'] = continue_data.goout.value_counts()[2]
num_goout_conti['3'] = continue_data.goout.value_counts()[3]
num_goout_conti['4'] = continue_data.goout.value_counts()[4]

prob_goout5_conti = (num_goout_conti['5']+1) / (len(continue_data)+5)
prob_goout1_conti = (num_goout_conti['1']+1) / (len(continue_data)+5)
prob_goout2_conti = (num_goout_conti['2']+1) / (len(continue_data)+5)
prob_goout3_conti = (num_goout_conti['3']+1) / (len(continue_data)+5)
prob_goout4_conti = (num_goout_conti['4']+1) / (len(continue_data)+5)

#find prob(health/continue)

num_health_conti = {}

num_health_conti['5'] = continue_data.health.value_counts()[5]
num_health_conti['1'] = continue_data.health.value_counts()[1]
num_health_conti['2'] = continue_data.health.value_counts()[2]
num_health_conti['3'] = continue_data.health.value_counts()[3]
num_health_conti['4'] = continue_data.health.value_counts()[4]

prob_health5_conti = (num_health_conti['5']+1) / (len(continue_data)+5)
prob_health1_conti = (num_health_conti['1']+1) / (len(continue_data)+5)
prob_health2_conti = (num_health_conti['2']+1) / (len(continue_data)+5)
prob_health3_conti = (num_health_conti['3']+1) / (len(continue_data)+5)
prob_health4_conti = (num_health_conti['4']+1) / (len(continue_data)+5)

#find prob(walc/continue)

num_walc_conti = {}

num_walc_conti['5'] = continue_data.Walc.value_counts()[5]
num_walc_conti['1'] = continue_data.Walc.value_counts()[1]
num_walc_conti['2'] = continue_data.Walc.value_counts()[2]
num_walc_conti['3'] = continue_data.Walc.value_counts()[3]
num_walc_conti['4'] = continue_data.Walc.value_counts()[4]

prob_walc5_conti = (num_walc_conti['5']+1) / (len(continue_data)+5)
prob_walc1_conti = (num_walc_conti['1']+1) / (len(continue_data)+5)
prob_walc2_conti = (num_walc_conti['2']+1) / (len(continue_data)+5)
prob_walc3_conti = (num_walc_conti['3']+1) / (len(continue_data)+5)
prob_walc4_conti = (num_walc_conti['4']+1) / (len(continue_data)+5)

#for drop data

#find prob(caste/drop)

num_caste_drop = {}

num_caste_drop['SC'] = drop_data.caste.value_counts()['SC']
num_caste_drop['ST'] = drop_data.caste.value_counts()['ST']
num_caste_drop['OC'] = drop_data.caste.value_counts()['OC']
num_caste_drop['BC'] = drop_data.caste.value_counts()['BC']

#print(num_caste_drop['SC'] + num_caste_drop['ST'] + num_caste_drop['OC'] + num_caste_drop['BC'])

prob_SC_g_drop = (num_caste_drop['SC']+1) / (len(drop_data)+4)
prob_ST_g_drop = (num_caste_drop['ST']+1) / (len(drop_data)+4)
prob_OC_g_drop = (num_caste_drop['OC']+1) / (len(drop_data)+4)
prob_BC_g_drop = (num_caste_drop['BC']+1) / (len(drop_data)+4)

#print(prob_SC_g_drop + prob_ST_g_drop + prob_OC_g_drop + prob_BC_g_drop)

#find prob(gender/drop)

num_gender_drop = {}

num_gender_drop['M'] = drop_data.gender.value_counts()[1]
num_gender_drop['F'] = drop_data.gender.value_counts()[0]

#print(num_gender_drop['M'] + num_gender_drop['F'])

prob_M_g_drop = (num_gender_drop['M']+1) / (len(drop_data)+2)
prob_F_g_drop = (num_gender_drop['F']+1) / (len(drop_data)+2)

#print(prob_M_g_drop + prob_F_g_drop)

#find prob(internet/drop)

num_internet_drop = {}

num_internet_drop['yes'] = drop_data.internet.value_counts()[1]
num_internet_drop['no'] = drop_data.internet.value_counts()[0]

#print(num_internet_drop['yes'] + num_internet_drop['no'])

prob_internet_drop = (num_internet_drop['yes']+1) / (len(drop_data)+2)
prob_notinternet_drop = (num_internet_drop['no']+1) / (len(drop_data)+2)

#print(prob_internet_drop + prob_notinternet_drop)


#find standard deviation for english / drop

english_drop = drop_data['english_marks'].values

mean_eng_drop = np.mean(english_drop)
eng_drop_std = standard_deviation(english_drop)


#find standard deviation for mathematics / drop

mathe_drop = drop_data['mathematics_marks'].values

mean_math_drop = np.mean(mathe_drop)
math_drop_std = standard_deviation(mathe_drop)


#find standard deviation for science / drop

science_drop = drop_data['science_marks'].values

mean_science_drop = np.mean(science_drop)
science_drop_std = standard_deviation(science_drop)

#find prob(guardian/drop)

num_guardian_drop = {}

num_guardian_drop['0'] = drop_data.guardian.value_counts()[0]
num_guardian_drop['1'] = drop_data.guardian.value_counts()[1]
num_guardian_drop['2'] = drop_data.guardian.value_counts()[2]

prob_mother_drop = (num_guardian_drop['0']+1) / (len(drop_data)+3)
prob_father_drop = (num_guardian_drop['1']+1) / (len(drop_data)+3)
prob_other_drop = (num_guardian_drop['2']+1) / (len(drop_data)+3)

#find prob(study/drop)

num_study_drop = {}

#num_study_drop['5'] = drop_data.studytime.value_counts()[5]
num_study_drop['1'] = drop_data.studytime.value_counts()[1]
num_study_drop['2'] = drop_data.studytime.value_counts()[2]
num_study_drop['3'] = drop_data.studytime.value_counts()[3]
num_study_drop['4'] = drop_data.studytime.value_counts()[4]


#prob_study0_drop = (num_study_drop['5']+1) / (len(drop_data)+5)
prob_study1_drop = (num_study_drop['1']+1) / (len(drop_data)+5)
prob_study2_drop = (num_study_drop['2']+1) / (len(drop_data)+5)
prob_study3_drop = (num_study_drop['3']+1) / (len(drop_data)+5)
prob_study4_drop = (num_study_drop['4']+1) / (len(drop_data)+5)

#find prob(failures/drop)

num_failures_drop = {}

num_failures_drop['0'] = drop_data.failures.value_counts()[0]
num_failures_drop['1'] = drop_data.failures.value_counts()[1]
num_failures_drop['2'] = drop_data.failures.value_counts()[2]
num_failures_drop['3'] = drop_data.failures.value_counts()[3]
#num_failures_drop['4'] = drop_data.failures.value_counts()[4]


prob_fail0_drop = (num_failures_drop['0']+1) / (len(drop_data)+5)
prob_fail1_drop = (num_failures_drop['1']+1) / (len(drop_data)+5)
prob_fail2_drop = (num_failures_drop['2']+1) / (len(drop_data)+5)
prob_fail3_drop = (num_failures_drop['3']+1) / (len(drop_data)+5)
#prob_fail4_drop = (num_failures_drop['4']+1) / (len(drop_data)+5)

#find prob(schoolsup/drop)

num_schoolsup_drop = {}

num_schoolsup_drop['yes'] = drop_data.schoolsup.value_counts()['yes']
num_schoolsup_drop['no'] = drop_data.schoolsup.value_counts()['no']

prob_schoolsup_drop = (num_schoolsup_drop['yes']+1) / (len(drop_data)+2)
prob_schoolnotsup_drop = (num_schoolsup_drop['no']+1) / (len(drop_data)+2)

#find prob(famsup/drop)

num_famsup_drop = {}

num_famsup_drop['yes'] = drop_data.famsup.value_counts()['yes']
num_famsup_drop['no'] = drop_data.famsup.value_counts()['no']

prob_famsup_drop = (num_famsup_drop['yes']+1) / (len(drop_data)+2)
prob_famnotsup_drop = (num_famsup_drop['no']+1) / (len(drop_data)+2)

#find standard deviation for absences / drop

absences_drop = drop_data['absences'].values

mean_absences_drop = np.mean(absences_drop)
absences_drop_std = standard_deviation(absences_drop)

#find prob(freetime/drop)

num_freetime_drop = {}

num_freetime_drop['5'] = drop_data.freetime.value_counts()[5]
num_freetime_drop['1'] = drop_data.freetime.value_counts()[1]
num_freetime_drop['2'] = drop_data.freetime.value_counts()[2]
num_freetime_drop['3'] = drop_data.freetime.value_counts()[3]
num_freetime_drop['4'] = drop_data.freetime.value_counts()[4]


prob_freetime5_drop = (num_freetime_drop['5']+1) / (len(drop_data)+5)
prob_freetime1_drop = (num_freetime_drop['1']+1) / (len(drop_data)+5)
prob_freetime2_drop = (num_freetime_drop['2']+1) / (len(drop_data)+5)
prob_freetime3_drop = (num_freetime_drop['3']+1) / (len(drop_data)+5)
prob_freetime4_drop = (num_freetime_drop['4']+1) / (len(drop_data)+5)

#find prob(goout/drop)

num_goout_drop = {}

num_goout_drop['5'] = drop_data.goout.value_counts()[5]
num_goout_drop['1'] = drop_data.goout.value_counts()[1]
num_goout_drop['2'] = drop_data.goout.value_counts()[2]
num_goout_drop['3'] = drop_data.goout.value_counts()[3]
num_goout_drop['4'] = drop_data.goout.value_counts()[4]

prob_goout5_drop = (num_goout_drop['5']+1) / (len(drop_data)+5)
prob_goout1_drop = (num_goout_drop['1']+1) / (len(drop_data)+5)
prob_goout2_drop = (num_goout_drop['2']+1) / (len(drop_data)+5)
prob_goout3_drop = (num_goout_drop['3']+1) / (len(drop_data)+5)
prob_goout4_drop = (num_goout_drop['4']+1) / (len(drop_data)+5)

#find prob(health/drop)

num_health_drop = {}

num_health_drop['5'] = drop_data.health.value_counts()[5]
num_health_drop['1'] = drop_data.health.value_counts()[1]
num_health_drop['2'] = drop_data.health.value_counts()[2]
num_health_drop['3'] = drop_data.health.value_counts()[3]
num_health_drop['4'] = drop_data.health.value_counts()[4]

prob_health5_drop = (num_health_drop['5']+1) / (len(drop_data)+5)
prob_health1_drop = (num_health_drop['1']+1) / (len(drop_data)+5)
prob_health2_drop = (num_health_drop['2']+1) / (len(drop_data)+5)
prob_health3_drop = (num_health_drop['3']+1) / (len(drop_data)+5)
prob_health4_drop = (num_health_drop['4']+1) / (len(drop_data)+5)

#find prob(walc/drop)

num_walc_drop = {}

num_walc_drop['5'] = drop_data.Walc.value_counts()[5]
num_walc_drop['1'] = drop_data.Walc.value_counts()[1]
num_walc_drop['2'] = drop_data.Walc.value_counts()[2]
num_walc_drop['3'] = drop_data.Walc.value_counts()[3]
num_walc_drop['4'] = drop_data.Walc.value_counts()[4]

prob_walc5_drop = (num_walc_drop['5']+1) / (len(drop_data)+5)
prob_walc1_drop = (num_walc_drop['1']+1) / (len(drop_data)+5)
prob_walc2_drop = (num_walc_drop['2']+1) / (len(drop_data)+5)
prob_walc3_drop = (num_walc_drop['3']+1) / (len(drop_data)+5)
prob_walc4_drop = (num_walc_drop['4']+1) / (len(drop_data)+5)

#Testing

predict = prediction(X_test)


#accuracy calculation
count = 0
for i in range(0,len(predict)):
    if(predict[i] == Y_test.iloc[i,0]):
        count = count+1

accuracy =  (count / len(predict)) * 100
print("Accuracy achieved =", accuracy, "%")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predict)
print()
print("Confusion Matrix")
print(cm)


TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

print()
print("classifier Predicted continue and actually it is continue")
print("True Positive = ", TP)
print("classifier predicted drop and actually it is drop")
print("True Negative = ", TN)
print("classifier predicted continue, but actually it is drop")
print("False Positive = ", FP)
print("classifier predicted drop, but actually it is continue")
print("False Negative = ", FN)

print()
print("Accuracy calculated using confusion matrix")
ac = (TP + TN) / len(predict)
print(ac)

print()
print("misclasification rate(error) : ")
er = (FP + FN) / len(predict)
print(er)

print()
print("Sensitivity: When the actual value is continue, how often is the prediction correct?")
TP_rate = TP / (FN + TP)
print("True positive rate = ", TP_rate)

print()
print("Specificity: When the actual value is drop, how often is the prediction correct?")
TN_rate = TN / (TN + FP)
print("True Negative rate = ", TN_rate)

print()
print("When the actual value is drop, how often is the prediction incorrect?")
FP_rate = FP / (TN + FP)
print("False positive rate = ", FP_rate)

print()
print("Precision: When a continue value is predicted, how often is the prediction correct?")
precision = TP / (TP + FP)
print("precision = ", precision)






