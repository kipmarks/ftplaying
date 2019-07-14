#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:42:52 2018

@author: kip
"""

# Automated Feature Engineering
# See https://www.kaggle.com/willkoehrsen/ ->
#          automated-feature-engineering-tutorial

# Required libraries
import pandas as pd
import numpy as np
import featuretools as ft
import warnings
warnings.filterwarnings('ignore')

# So we have the entities and 2 types of transaction
# clients -< loans -< payments
clients = pd.read_csv('/media/veracrypt1/Kip/Projects/44_Ird/featuretools/input/clients.csv',parse_dates=['joined'])
loans = pd.read_csv('/media/veracrypt1/Kip/Projects/44_Ird/featuretools/input/loans.csv',parse_dates=['loan_start','loan_end'])
payments = pd.read_csv('/media/veracrypt1/Kip/Projects/44_Ird/featuretools/input/payments.csv',parse_dates=['payment_date'])

# Rollup payments to loan_id then merge into loans
stats = payments.groupby('loan_id')['payment_amount','missed'].agg(['sum'])
stats.columns = ['payment','missed']
loans2 = loans.merge(stats, left_on='loan_id', right_index=True, how='left')
loans2.head()

# Now rollup to client id and merge into clients

stats = loans2.groupby('client_id')['loan_amount','payment','missed'].agg(['sum'])
stats.columns = ['loan_amount','payment','missed']
clients2 = clients.merge(stats, left_on='client_id', right_index=True, how='left')

# Then add some features manually
clients2['join_month'] = clients2['joined'].dt.month
clients2['log_income'] = np.log(clients2['income'])

# 5 new features with 7 lines of code
clients2.head()
##########################################################################
# So far so good.
# OK Let's use featuretools
# 
# An ft entity is simply a data-frame. And ft uses sets of them - an entityset!
# Basically we're creating metadata. NB payments has no payment_id, so create one.
es = ft.EntitySet(id = 'myentityset')
es = es.entity_from_dataframe(entity_id = 'clients', 
                              dataframe=clients, 
                              index='client_id',
                              time_index='joined')

es = es.entity_from_dataframe(entity_id = 'loans', 
                              dataframe=loans, 
                              variable_types = {'repaid': ft.variable_types.Categorical},
                              index='loan_id',
                              time_index='loan_start')

es = es.entity_from_dataframe(entity_id = 'payments', 
                              dataframe=payments, 
                              variable_types = {'missed': ft.variable_types.Categorical},
                              make_index = True,
                              index='payment_id',
                              time_index='payment_date')

# Right, lets have a look. Its just metadata in a dict
es['loans']

# Now we have to tell ft how these entities are related
# Uses the parent-child metaphor. Create the relationships then add to  the entity set
r_c_l = ft.Relationship(es['clients']['client_id'],
                        es['loans']['client_id'])
r_l_p = ft.Relationship(es['loans']['loan_id'],
                        es['payments']['loan_id'])
es = es.add_relationship(r_c_l)
es = es.add_relationship(r_l_p)

# Another look at the whole lot
es

# Ok Now lets make some features. First some primitives
# Either aggregations or transformations
primitives = ft.list_primitives()
pd.options.display.max_colwidth=100
primitives[primitives['type']=='aggregation'].head(20)

# And
primitives[primitives['type']=='transform'].head(20)


# Ok lets do it! Make some features for the clients
features, feature_names = ft.dfs(entityset = es,
                                 target_entity='clients',
                                 agg_primitives=['median','mean','std','max','percent_true','last','time_since_last'],
                                 trans_primitives=['years','month','divide'])

# Wow! After setup (which could be done in a library call)
# we have 408  features with 4 lines of code (140 new ones)
# I'm already sold on this!
len(feature_names)

feature_names



# Some interesting column names!
# Here is a feature of depth=1 ie only 1 layer of primitives has been used
pd.DataFrame(features['MEAN(payments.payment_amount)'].head())

# A feature of depth=2
pd.DataFrame(features['LAST(loans.MEAN(payments.payment_amount))'].head())


# Number of features  increases to 3530 with depth=3
features, feature_names = ft.dfs(entityset = es,
                                 target_entity='clients',
                                 agg_primitives=['median','mean','std','max','percent_true','last','time_since_last'],
                                 trans_primitives=['years','month','divide'],
                                 max_depth=3)
len(feature_names)
feature_names