
# coding: utf-8

# In[1]:


import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab
import scipy.stats as st

from matplotlib.ticker import PercentFormatter
from matplotlib import colors
from statsmodels.distributions.empirical_distribution import ECDF
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


csv.register_dialect('spaceDialect', delimiter = ' ')

dados_medicos = []

with open('D:/UFRJ/Estatistica e Modelos Probabilisticos/trabalho/Dados-medicos.csv', 'r') as f:
    reader = csv.reader(f, dialect='spaceDialect')
    for row in reader:
        filtered = []
        # eliminating empty strings from the dataset
        for i in row:
            if i:
                # converting to float when possible
                try:
                    filtered.append(float(i))
                except:
                    continue
        dados_medicos.append(filtered)

dados_medicos = np.array(dados_medicos[1:])


# In[3]:


# calculation of the mean and variance
def mean_variance_std(data, index):
    new_data = []
    for row in data:
        new_data.append(row[index])
    mean = float(sum(new_data)) / max(len(new_data), 1)
    variance = np.var(new_data)
    result = [mean, variance, np.std(new_data)]
    return result


# In[4]:


theoretical_bin_amount = (1+(3.3)*(np.log10(1172)))

def theoretical_bin_size(index):
    return ((3.49)*(mean_variance_std(dados_medicos,index)[2])/(np.cbrt(1172)))


# In[5]:


def simple_histogram(data, index, color = 'green', label='Simple Histogram'):
    bin_size = theoretical_bin_size(index)
    new_data = []
    for row in data:
        new_data.append(row[index])
    
    n_bins = range( int(min(new_data)), int(max(new_data)), int(bin_size))
    print('Número de bins:',len(n_bins))
    
    plt.hist(new_data, bins=n_bins, color=color, label=label, normed=True)
    plt.title(label)
    plt.ylabel('Frequency')


# ## 2.1 Histograma e Funícção Distribuição Empírica

# In[6]:


simple_histogram(dados_medicos, 0, color='black', label='Idade (anos)')


# In[7]:


simple_histogram(dados_medicos, 1, 'red', 'Peso (kg)')


# In[8]:


simple_histogram(dados_medicos, 2, 'green', 'Carga Final (Watt)')


# In[9]:


simple_histogram(dados_medicos, 3, 'blue', 'VO2 máx (mL/kg/min)')


# In[10]:


def ecdf(data, index, color, label):
    new_data = []
    for row in data:
        new_data.append(row[index])
    ecdf = ECDF(new_data)
    plt.plot(ecdf.x, ecdf.y, color, label, marker=".")
    plt.show()


# In[11]:


ecdf(dados_medicos, 0, 'black', 'Idade (anos)')


# In[12]:


ecdf(dados_medicos, 1, 'red', 'Peso (kg)')


# In[13]:


ecdf(dados_medicos, 2, 'green', 'Carga Final (Watt)')


# In[14]:


ecdf(dados_medicos, 3, 'blue', 'VO2 máx (mL/kg/min)')


# In[15]:


def simple_boxplot(data, index, label='Simple Boxplot'):
    new_data = []
    for row in data:
        new_data.append(row[index])
    plt.boxplot(new_data, 0, 'gD', label)


# ## 2.2 Média, Variância e Boxplot

# In[16]:


print('Média, Variância e Desvio Padrão de Idade:')
print(mean_variance_std(dados_medicos,0))
print('\n')
print('Média, Variância e Desvio Padrão de Peso:')
print(mean_variance_std(dados_medicos,1))
print('\n')
print('Média, Variância e Desvio Padrão de Carga Final:')
print(mean_variance_std(dados_medicos,2))
print('\n')
print('Média, Variância e Desvio Padrão de VO2 máximo:')
print((mean_variance_std(dados_medicos,3)))


# In[17]:


simple_boxplot(dados_medicos, 0, 'Idade (anos)')


# In[18]:


simple_boxplot(dados_medicos, 1, 'Peso (kg)')


# In[19]:


simple_boxplot(dados_medicos, 2, 'Carga Final (Watt)')


# In[20]:


simple_boxplot(dados_medicos, 3, 'VO2 máx (mL/kg/min)')


# ## 2.3 Parametrizando distribuições

# In[21]:


def fit_distribution(data, index, distribution):
    labels=['Idade', 'Peso', 'Carga', 'VO2']
    data = pd.DataFrame.from_records(data, columns=labels)
        
    data = distribution.fit(data[labels[index]])
    return data


# In[22]:


dist_idade = []
dist_idade.append(fit_distribution(dados_medicos, 0, st.norm))
dist_idade.append(fit_distribution(dados_medicos, 0, st.expon))
dist_idade.append(fit_distribution(dados_medicos, 0, st.lognorm))
dist_idade.append(fit_distribution(dados_medicos, 0, st.weibull_min))


# In[23]:


print(dist_idade)


# In[24]:


#def distribution(ax, distribution, shape=None, scale, loc):
#    x_axis = range(int(min(new_data)), int(max(new_data)))
#    y_axis = []
#    
#    if shape == None:
#        y_axis.append(distribution.pdf(i, loc, scale))
#    else:
#        y_axis.append(distribution.pdf(i, shape, loc, scale))


# In[25]:


def distribution_histogram(data, dist_data, index, dist_index, label, distribution, color_h, color_d):
    shape = 0.0
    loc = 0.0
    scale = 0.0
    new_data = []
    for row in data:
        new_data.append(row[index])
        
    x_axis = range(int(min(new_data)), int(max(new_data)))
    x_axis = [float(i) for i in x_axis]
    y_axis = []
    
    if len(list(dist_data[dist_index])) == 3:
        shape = dist_data[dist_index][0]
        loc = dist_data[dist_index][1]
        scale = dist_data[dist_index][2]
        
        for i in x_axis:
            y_axis.append(distribution.pdf(i, shape, loc, scale))
    else:
        loc = dist_data[dist_index][0]
        scale = dist_data[dist_index][1]
        
        for i in x_axis:
            y_axis.append(distribution.pdf(i, loc, scale))
    
        
    plt.plot(x_axis, y_axis, color=color_d, label='None', marker=".")
    simple_histogram(dados_medicos, index, color=color_h, label=label)


# In[26]:


distribution_histogram(dados_medicos, dist_idade, 0, 0, 'Idade (anos)', st.norm, 'black', 'green')
plt.show()
distribution_histogram(dados_medicos, dist_idade, 0, 1, 'Idade (anos)', st.expon, 'black', 'green')
plt.show()
distribution_histogram(dados_medicos, dist_idade, 0, 2, 'Idade (anos)', st.lognorm, 'black', 'green')
plt.show()
distribution_histogram(dados_medicos, dist_idade, 0, 3, 'Idade (anos)', st.weibull_min, 'black', 'green')


# In[27]:


dist_peso = []
dist_peso.append(fit_distribution(dados_medicos, 1, st.norm))
dist_peso.append(fit_distribution(dados_medicos, 1, st.expon))
dist_peso.append(fit_distribution(dados_medicos, 1, st.lognorm))
dist_peso.append(fit_distribution(dados_medicos, 1, st.weibull_min))
print(dist_peso)


# In[28]:


distribution_histogram(dados_medicos, dist_peso, 1, 0, 'Peso (kg)', st.norm, 'red', 'black')
plt.show()
distribution_histogram(dados_medicos, dist_peso, 1, 1, 'Peso (kg)', st.expon, 'red', 'black')
plt.show()
distribution_histogram(dados_medicos, dist_peso, 1, 2, 'Peso (kg)', st.lognorm, 'red', 'black')
plt.show()
distribution_histogram(dados_medicos, dist_peso, 1, 3, 'Peso (kg)', st.weibull_min, 'red', 'black')


# In[29]:


dist_carga = []
dist_carga.append(fit_distribution(dados_medicos, 2, st.norm))
dist_carga.append(fit_distribution(dados_medicos, 2, st.expon))
dist_carga.append(fit_distribution(dados_medicos, 2, st.lognorm))
dist_carga.append(fit_distribution(dados_medicos, 2, st.weibull_min))
print(dist_carga)


# In[30]:


distribution_histogram(dados_medicos, dist_carga, 2, 0, 'Carga final (watt)', st.norm, 'green', 'black')
plt.show()
distribution_histogram(dados_medicos, dist_carga, 2, 1, 'Carga final (watt)', st.expon, 'green', 'black')
plt.show()
distribution_histogram(dados_medicos, dist_carga, 2, 2, 'Carga final (watt)', st.lognorm, 'green', 'black')
plt.show()
distribution_histogram(dados_medicos, dist_carga, 2, 3, 'Carga final (watt)', st.weibull_min, 'green', 'black')


# In[31]:


dist_vo = []
dist_vo.append(fit_distribution(dados_medicos, 3, st.norm))
dist_vo.append(fit_distribution(dados_medicos, 3, st.expon))
dist_vo.append(fit_distribution(dados_medicos, 3, st.lognorm))
dist_vo.append(fit_distribution(dados_medicos, 3, st.weibull_min))
print(dist_vo)


# In[32]:


distribution_histogram(dados_medicos, dist_vo, 3, 0, 'VO2 máx (mL/kg/min)', st.norm, 'blue', 'black')
plt.show()
distribution_histogram(dados_medicos, dist_vo, 3, 1, 'VO2 máx (mL/kg/min)', st.expon, 'blue', 'black')
plt.show()
distribution_histogram(dados_medicos, dist_vo, 3, 2, 'VO2 máx (mL/kg/min)', st.lognorm, 'blue', 'black')
plt.show()
distribution_histogram(dados_medicos, dist_vo, 3, 3, 'VO2 máx (mL/kg/min)', st.weibull_min, 'blue', 'black')


# ## 2.4 Gráfico QQplot ou ProbabilityPlot

# In[33]:


def prob_plot(data, dist_data, index, dist_index, distribution):       
    shape = 0.0
    loc = 0.0
    scale = 0.0
    new_data = []
    for row in data:
        new_data.append(row[index])
        
    x_axis = range(int(min(new_data)), int(max(new_data)))
    x_axis = [float(i) for i in x_axis]
    y_axis = []
    
    if len(list(dist_data[dist_index])) == 3:
        shape = dist_data[dist_index][0]
        loc = dist_data[dist_index][1]
        scale = dist_data[dist_index][2]
        
        for i in x_axis:
            y_axis.append(distribution.pdf(i, shape, loc, scale))
    else:
        loc = dist_data[dist_index][0]
        scale = dist_data[dist_index][1]
        
        for i in x_axis:
            y_axis.append(distribution.pdf(i, loc, scale))
    
    st.probplot(new_data, sparams = dist_data[dist_index], dist=distribution, plot=pylab)
    pylab.show()


# ### Idade

# In[34]:


prob_plot(dados_medicos, dist_idade, 0, 0, st.norm)
prob_plot(dados_medicos, dist_idade, 0, 1, st.expon)
prob_plot(dados_medicos, dist_idade, 0, 2, st.lognorm)
prob_plot(dados_medicos, dist_idade, 0, 3, st.weibull_min)


# ### Peso

# In[35]:


prob_plot(dados_medicos, dist_peso, 1, 0, st.norm)
prob_plot(dados_medicos, dist_peso, 1, 1, st.expon)
prob_plot(dados_medicos, dist_peso, 1, 2, st.lognorm)
prob_plot(dados_medicos, dist_peso, 1, 3, st.weibull_min)


# ### Carga final

# In[36]:


prob_plot(dados_medicos, dist_carga, 2, 0, st.norm)
prob_plot(dados_medicos, dist_carga, 2, 1, st.expon)
prob_plot(dados_medicos, dist_carga, 2, 2, st.lognorm)
prob_plot(dados_medicos, dist_carga, 2, 3, st.weibull_min)


# ### VO2 máximo

# In[37]:


prob_plot(dados_medicos, dist_vo, 3, 0, st.norm)
prob_plot(dados_medicos, dist_vo, 3, 1, st.expon)
prob_plot(dados_medicos, dist_vo, 3, 2, st.lognorm)
prob_plot(dados_medicos, dist_vo, 3, 3, st.weibull_min)


# ## 2.5 Teste de Hipótese

# In[38]:


def komolgorov_smirnov(data, dist_data, index, dist_index, distribution):
    shape = 0.0
    loc = 0.0
    scale = 0.0
    
    args=()
    
    new_data = []
    for row in data:
        new_data.append(row[index])
    
    if len(list(dist_data[dist_index])) == 3:
        shape = dist_data[dist_index][0]
        loc = dist_data[dist_index][1]
        scale = dist_data[dist_index][2]
        
        args = (shape, loc, scale)

    else:
        loc = dist_data[dist_index][0]
        scale = dist_data[dist_index][1]
        
        args = (loc, scale)
    
    aux = st.kstest(new_data, distribution.cdf, args)
    return aux


# In[39]:


print(komolgorov_smirnov(dados_medicos, dist_idade, 0, 0, st.norm))
print(komolgorov_smirnov(dados_medicos, dist_idade, 0, 1, st.expon))
print(komolgorov_smirnov(dados_medicos, dist_idade, 0, 2, st.lognorm))
print(komolgorov_smirnov(dados_medicos, dist_idade, 0, 3, st.weibull_min))


# In[40]:


print(komolgorov_smirnov(dados_medicos, dist_peso, 1, 0, st.norm))
print(komolgorov_smirnov(dados_medicos, dist_peso, 1, 1, st.expon))
print(komolgorov_smirnov(dados_medicos, dist_peso, 1, 2, st.lognorm))
print(komolgorov_smirnov(dados_medicos, dist_peso, 1, 3, st.weibull_min))


# In[41]:


print(komolgorov_smirnov(dados_medicos, dist_carga, 2, 0, st.norm))
print(komolgorov_smirnov(dados_medicos, dist_carga, 2, 1, st.expon))
print(komolgorov_smirnov(dados_medicos, dist_carga, 2, 2, st.lognorm))
print(komolgorov_smirnov(dados_medicos, dist_carga, 2, 3, st.weibull_min))


# In[42]:


print(komolgorov_smirnov(dados_medicos, dist_vo, 3, 0, st.norm))
print(komolgorov_smirnov(dados_medicos, dist_vo, 3, 1, st.expon))
print(komolgorov_smirnov(dados_medicos, dist_vo, 3, 2, st.lognorm))
print(komolgorov_smirnov(dados_medicos, dist_vo, 3, 3, st.weibull_min))


# ## 2.6 Análise de dependência entre as variáveis, modelo de regressão

# In[43]:


def simple_pearsonr(data, index_x, index_y):
    x_data = []
    y_data = []
    
    for row in data:
        x_data.append(row[index_x])
        y_data.append(row[index_y])
        
    return st.pearsonr(x_data, y_data)


# In[44]:


print(simple_pearsonr(dados_medicos, 0, 3))
print(simple_pearsonr(dados_medicos, 1, 3))
print(simple_pearsonr(dados_medicos, 2, 3))


# In[45]:


def dependency_to_vo2(input_data, ax, index, y_label, color):
    vo2_column_data = [i[3] for i in input_data]
    y_column_data = [i[index] for i in input_data]
    
    linear_regression = st.linregress(vo2_column_data, y_column_data)
    max_vo2 = max(vo2_column_data)
    slope = linear_regression[0]
    intercept = linear_regression[1]
    ax.plot([0.0, max_vo2], [intercept, slope*max_vo2 + intercept], color=color)
    
    ax.scatter(vo2_column_data, y_column_data)
    ax.set_xlabel(u'VO2')
    ax.set_ylabel(y_label)
    
    ax.set_title(u'Pearson coeffiecient: %s' % round(linear_regression[2], 3))


# In[46]:


fig, axs = plt.subplots(1, 3, figsize=(15,5))
dependency_to_vo2(dados_medicos, axs[0], index=0, y_label=u'Idade', color='black')
dependency_to_vo2(dados_medicos, axs[1], index=1, y_label=u'Peso', color='red')
dependency_to_vo2(dados_medicos, axs[2], index=2, y_label=u'Carga final', color='green')


# ## 2.7 Inferência Bayesiana

# In[47]:


# H A: VO2 máx < 35
# H B: 35 =< VO2 máx

# H1: Carga final < 100
# H2: 100 =< Carga final <= 200
# H3: 200 =< Carga final <= 300
# H4: 300 =< Carga final


# In[48]:


def priori(data, intervalo, index): 
    new_data = []
    for row in data:
        new_data.append(row[index])
    
    subtotal = 0
    total = len(new_data)
    
    for i in new_data:
        if min(intervalo)<=i and i<max(intervalo):
            subtotal+=1
    
    return float(subtotal/total)


#P(H_A|H1)
def likelihood(data, intervalo_H, intervalo_HA, index):

    prior = [] # H1
    
    for i in data:
        if min(intervalo_H)<=i[index] and i[index]<max(intervalo_H):
            prior.append(i)
    
    sub_prior = [] # HA
    
    for i in prior:
        #print(i, min(intervalo_HA), max(intervalo_HA))
        if min(intervalo_HA)<=i[3] and i[3]<max(intervalo_HA):
            sub_prior.append(i)
            
    return len(sub_prior)/len(prior)


def posteriori(priori_list, likelihood_list, bayes_index):
    
    bayes_numerators = []
    
    for i in range(len(priori_list)):
            bayes_numerators.append(priori_list[i]*likelihood_list[i])
    print(bayes_numerators)
    return (bayes_numerators[bayes_index])/(sum(bayes_numerators))
    


# In[49]:


H_A = [0, 35]
H_B = [35, 80]

H_1 = [  0, 100]
H_2 = [100, 200]
H_3 = [200, 300]
H_4 = [300, 400]


# In[50]:


print(priori(dados_medicos, H_1, 2))
print(priori(dados_medicos, H_2, 2))
print(priori(dados_medicos, H_3, 2))
print(priori(dados_medicos, H_4, 2))


# In[51]:


print(likelihood(dados_medicos, H_1, H_A, 2))
print(likelihood(dados_medicos, H_2, H_A, 2))
print(likelihood(dados_medicos, H_3, H_A, 2))
print(likelihood(dados_medicos, H_4, H_A, 2))


# In[52]:


print(likelihood(dados_medicos, H_1, H_B, 2))
print(likelihood(dados_medicos, H_2, H_B, 2))
print(likelihood(dados_medicos, H_3, H_B, 2))
print(likelihood(dados_medicos, H_4, H_B, 2))


# In[53]:


p_list = [priori(dados_medicos, H_1, 2), priori(dados_medicos, H_2, 2),
              priori(dados_medicos, H_3, 2), priori(dados_medicos, H_4, 2)]
l_list = [likelihood(dados_medicos, H_1, H_A, 2), likelihood(dados_medicos, H_2, H_A, 2),
                  likelihood(dados_medicos, H_2, H_A, 2), likelihood(dados_medicos, H_4, H_A, 2)]


# In[54]:


print(posteriori(p_list, l_list, 0))
print(posteriori(p_list, l_list, 1))
print(posteriori(p_list, l_list, 2))
print(posteriori(p_list, l_list, 3))


# In[55]:


p_list = [priori(dados_medicos, H_1, 2), priori(dados_medicos, H_2, 2),
              priori(dados_medicos, H_3, 2), priori(dados_medicos, H_4, 2)]
l_list = [likelihood(dados_medicos, H_1, H_B, 2), likelihood(dados_medicos, H_2, H_B, 2),
                  likelihood(dados_medicos, H_2, H_B, 2), likelihood(dados_medicos, H_4, H_B, 2)]


# In[56]:


print(posteriori(p_list, l_list, 0))
print(posteriori(p_list, l_list, 1))
print(posteriori(p_list, l_list, 2))
print(posteriori(p_list, l_list, 3))

