#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[4]:


dataframe.describe().T


# In[5]:


norm = dataframe['normal']
binom = dataframe['binomial']


# In[6]:


infos = pd.DataFrame(data={
    'normal': [
        norm.mode()[0]
        , norm.mean()
        , norm.median()
    ],
    
    'binomial': [
        binom.mode()[0]
        , binom.mean()
        , binom.median()
    ]
}, index=['Moda', 'Media', 'Mediana'])


# In[7]:


infos


# In[8]:


#fig = plt.figure()

#ax1 = fig.add_subplot(221)
#ax2 = fig.add_subplot(222)

#sns.distplot(norm, ax=ax1)
#ax1.set_title('Distribuição normal')

#sns.distplot(binom, ax=ax2)
#ax2.set_title('Distribuição binomial')

#plt.show()


# In[9]:


#fig2 = plt.figure()

#ax3 = fig2.add_subplot(221)
#ax4 = fig2.add_subplot(222)

#ec_no = ECDF(norm)
#norm_cdf = ec_no(norm)

#ec_bi = ECDF(binom)
#binom_cdf = ec_bi(binom)

#sns.lineplot(norm, norm_cdf, ax=ax3)
#ax3.set_title('CDF - Distribuição Normal')

#sns.lineplot(binom, binom_cdf, ax=ax4)
#ax4.set_title('CDF - Distribuição Binomial')

#plt.show()


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[10]:


q1_norm = (norm.quantile(q=.25) - binom.quantile(q=.25)).round(3)
q2_norm = (norm.quantile(q=.5) - binom.quantile(q=.5)).round(3)
q3_norm = (norm.quantile(q=.75) - binom.quantile(q=.75)).round(3)


# In[11]:


def q1():
    return (q1_norm, q2_norm, q3_norm)
q1()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[12]:


ecdf = ECDF(norm)
t1 = ecdf((norm.mean() + norm.std()))
t2 = ecdf((norm.mean() - norm.std()))


# In[13]:


def q2():
    return float(np.round((t1 - t2), 3))
q2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# In[14]:


ecdf = ECDF(norm)

t3 = ecdf((norm.mean() + 2 * norm.std()))
t4 = ecdf((norm.mean() - 2 * norm.std()))

t5 = ecdf((norm.mean() + 3 * norm.std()))
t6 = ecdf((norm.mean() - 3 * norm.std()))

((t3 - t4).round(3), (t5 - t6).round(3))


# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[15]:


m_norm = norm.mean()
v_norm = norm.var()

m_binom = binom.mean()
v_binom = binom.var()


# In[16]:


def q3():
    return (np.round((m_binom - m_norm), 3), np.round((v_binom - v_norm), 3))
q3()


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[17]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[18]:


# Sua análise da parte 2 começa aqui.
stars.sample(5)


# In[19]:


stars.describe().T


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[20]:


false_pulsar_mean_profile = stars[stars['target'] == False]['mean_profile']
false_pulsar_mean_profile_standardized = false_pulsar_mean_profile.apply(lambda x: (x - np.mean(false_pulsar_mean_profile)) / np.std(false_pulsar_mean_profile))


# In[21]:


qtl80, qtl90, qtl95 = sct.norm.ppf(q=[.8, .9, .95], loc=0, scale=1)


# In[22]:


ecdf_pulsar = ECDF(false_pulsar_mean_profile_standardized)


# In[23]:


def q4():
    return (np.round(ecdf_pulsar(qtl80), 3), np.round(ecdf_pulsar(qtl90), 3), np.round(ecdf_pulsar(qtl95), 3))
q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[24]:


qtl25_scaler, qtl50_scaler, qtl75_scaler = sct.norm.ppf(q=[.25, .5, .75], loc=0, scale=1)
qtl25 = np.quantile(a=false_pulsar_mean_profile_standardized, q=.25)
qtl50 = np.quantile(a=false_pulsar_mean_profile_standardized, q=.5)
qtl75 = np.quantile(a=false_pulsar_mean_profile_standardized, q=.75)


# In[25]:


def q5():
    return (np.round((qtl25 - qtl25_scaler), 3), np.round((qtl50 - qtl50_scaler), 3), np.round((qtl75 - qtl75_scaler), 3))
q5()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
