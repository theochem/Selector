import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import mean_absolute_error,mean_squared_error,pairwise_distances,max_error
from DiverseSelector.methods.dissimilarity import MaxMin, OptiSim


# define potential energy function
def potential_energy(q1,q2):
    '''
    q1: float or np.array

    q2: float or np.array

    '''
    v_0 = 5.0
    a_0 = 0.6
    a = [3.0,1.5,3.2,2.0]
    b_q1 = 0.1
    b_q2 = 0.1
    sigma_q1 = [0.3,1.0,0.4,1.0]
    sigma_q2 = [0.4,1.0,1.0,0.1]
    alpha = [1.3,-1.5,1.4,-1.3]
    beta = [-1.6,-1.7,1.8,1.23]

    v = v_0 + a_0*(np.exp(-(q1-b_q1)**2-(q2-b_q2)**2))
    
    for i in range(4):
        v = v - a[i]*np.exp(-sigma_q1[i]*(q1-alpha[i])**2-sigma_q2[i]*(q2-beta[i])**2)
    
    return v


# define boltzmann method
def boltzmann_sample(database,k):
    '''
        database : np.array
            the database we want to sample

        k : np.array or int, float
            when k is larger, the database is more unbiased
            the order of 1.5
    '''

    E_0 = 1.780     # the minimum energy

    # if k == int
    if type(k) == int or type(k) == float:
        sample_list = []

        rng = np.random.default_rng()
        rn = rng.random((len(database),))
        p = np.exp(-np.array(k)*(1.5)*((potential_energy(database[:,0],database[:,1]))-E_0))
        for i in range(len(database)):
            if p[i] >= rn[i]:
                sample_list.append(database[i])

        sample_list = np.array(sample_list)
        
        return sample_list
    
    # if k == np.array
    sample_list = {}

    for i in range(len(k)):
        sample_list[str(k[i])] = []
    
    for j in range(len(k)):      # if all of the boltzmann sampling databases have the sample_number points then break out
        rng = np.random.default_rng()       # define a random data generate tool
        rn = rng.random((len(database),1))       # generate len(k) random numbers
        p = np.exp(-np.array(k[j])*(1.5)*((potential_energy(database[:,0],database[:,1]))-E_0))     # calculate the probability of the point from -3 to 3
        for i in range(len(database)):            
            if  p[i] >= rn[i]:        # if p >= random_number than choose the point
                sample_list[str(k[j])].append(database[i])

    for i in range(len(k)):
        sample_list[str(k[i])]=np.array(sample_list[str(k[i])])
 
    return sample_list


# define random sample method
def random_sample(data,sample_number,sample_times):
    sample_list=[]
    for sample_time in range(sample_times):
        label = np.random.choice(np.arange(len(data)),sample_number,replace=False)
        sample_list.append(data[label])
    
    sample_list = np.array(sample_list)
    return sample_list


def generate_data(number):
    q1 = np.linspace(-3,3,number)    
    q2 = np.linspace(-3,3,number)

    data = []
    for i in range(len(q1)):
        for j in range(len(q2)):
            data.append([q1[i],q2[j]])

    data = np.array(data)
    
    return data


# define a class of error calculator
class ErrorCal():
    '''
    ----------
    features: np.ndarray
        Indices of points' number and features
        row of features represents the number of points
        column of features represents the number of points' features

    values: np.ndarray
        values of each points

    sample_number: int
        The maximum number of selected points
    '''
    def __init__(self,features,values,sample_number,function=None,labels=None):
        self.features = features
        self.values = values
        self.sample_number = sample_number
        self.function = function
        self.labels = labels


    # define random_error calculator
    def random_error(self):
        # choose points with random sampling
        points_number = np.arange(len(self.features))       #the total points number
        label = np.random.choice(points_number,self.sample_number,replace=False)     #random choose the label of points
        features_choice = self.features[label]       #the selected points'features
        values_choice = self.values[label]       #the selected points'values

        # fit with Gaussian Process
        kernel = ConstantKernel(constant_value=0.2) + RBF(length_scale=0.5, length_scale_bounds=(1e-7, 1e+7))
        gpr = GaussianProcessRegressor(n_restarts_optimizer=25,kernel=kernel)
        model = gpr.fit(features_choice,values_choice)

         # test the error by regular grid method
        test_points = generate_data(100)

        # calculate the maximum error
        random_m_error = max_error(model.predict(test_points),potential_energy(test_points[:,0],test_points[:,1]))
        # calculate the mean squared error
        random_ms_error = mean_squared_error(model.predict(test_points),potential_energy(test_points[:,0],test_points[:,1]))
        # calculate the mean absolute error   
        random_ma_error = mean_absolute_error(model.predict(test_points),potential_energy(test_points[:,0],test_points[:,1]))        

        return random_m_error,random_ms_error,random_ma_error


    # define maxmin_error calculator
    def maxmin_error(self):
        # choose points with maxmin sampling
        if self.function != None:
            features_dist = self.function(self.features)
        else:
            features_dist = pairwise_distances(X=self.features,Y=None,metric="euclidean")

        selector = MaxMin()
        selected_ids = selector.select(arr=features_dist,size=self.sample_number,labels=self.labels)
        features_choice = self.features[selected_ids]
        values_choice = self.values[selected_ids]
        
        # fit with Gaussian Process
        kernel = ConstantKernel(constant_value=0.2) + RBF(length_scale=0.5, length_scale_bounds=(1e-7, 1e+7))
        gpr = GaussianProcessRegressor(n_restarts_optimizer=25,kernel=kernel)
        model = gpr.fit(features_choice,values_choice)

        # test the error by regular grid method
        test_points = generate_data(100)

        # calculate the maximum error
        maxmin_m_error = max_error(model.predict(test_points),potential_energy(test_points[:,0],test_points[:,1]))
        # calculate the mean squared error
        maxmin_ms_error = mean_squared_error(model.predict(test_points),potential_energy(test_points[:,0],test_points[:,1]))
        # calculate the mean absolute error   
        maxmin_ma_error = mean_absolute_error(model.predict(test_points),potential_energy(test_points[:,0],test_points[:,1])) 

        return maxmin_m_error,maxmin_ms_error,maxmin_ma_error   


    # define optisim_error calculator
    def optisim_error(self):
        # choose points with optisim sampling
        selector = OptiSim()       #the total points number
        selected_ids = selector.select(self.features,size=self.sample_number,labels=self.labels)
        features_choice = self.features[selected_ids]       #the selected points'features
        values_choice = self.values[selected_ids]       #the selected points'values

        # fit with Gaussian Process
        kernel = ConstantKernel(constant_value=0.2) + RBF(length_scale=0.5, length_scale_bounds=(1e-7, 1e+7))
        gpr = GaussianProcessRegressor(n_restarts_optimizer=25,kernel=kernel)
        model = gpr.fit(features_choice,values_choice)

        # test the error by regular grid method
        test_points = generate_data(100)
        
        # calculate the maximum error
        optisim_m_error = max_error(model.predict(test_points),potential_energy(test_points[:,0],test_points[:,1]))
        # calculate the mean squared error
        optisim_ms_error = mean_squared_error(model.predict(test_points),potential_energy(test_points[:,0],test_points[:,1]))
        # calculate the mean absolute error   
        optisim_ma_error = mean_absolute_error(model.predict(test_points),potential_energy(test_points[:,0],test_points[:,1])) 

        return optisim_m_error,optisim_ms_error,optisim_ma_error    
    
if __name__ == '__main__':
    # generate a total random database
    rng = np.random.default_rng(seed=42)
    total_database = (rng.random((int(1e7),2))-0.5)*6


    # using boltzmann sampling to get a sub_database
    k = np.arange(1,5)
    # k = np.append(k,np.inf)

    sub_database = boltzmann_sample(total_database,k)


    # using random sampling in the sub_database to get 1e3 points sub_sub_database
    sample_times = 30
    sample_number = 1000
    sub_sub_database = {}
    for i in range(len(k)):
        sub_sub_database[str(k[i])] = random_sample(sub_database[str(k[i])],sample_number,sample_times)


    selected_number = range(1,1000,1)

    random_error = {}   # maximum error, mean squared error, mean absolute error 
    maxmin_error = {}   # maximum error, mean squared error, mean absolute error 
    optisim_error = {}  # maximum error, mean squared error, mean absolute error 

    for i in range(len(k)):     # iteration with different k
        random_error[str(k[i])] = []
        maxmin_error[str(k[i])] = []
        optisim_error[str(k[i])] = []   

        for s in selected_number:        # the number of selected_points
            random_error_list = []
            maxmin_error_list = []
            optsim_error_list = []

            for j in range(sample_times):
                selector = ErrorCal(sub_sub_database[str(k[i])][j,:],potential_energy(sub_sub_database[str(k[i])][j,:][:,0],
                                                                                    sub_sub_database[str(k[i])][j,:][:,1]),s)
                random_error_list.append(selector.random_error())
                maxmin_error_list.append(selector.maxmin_error())
                optsim_error_list.append(selector.optisim_error())

            random_error_list = np.mean(random_error_list,axis=0)
            random_error[str(k[i])].append(random_error_list)

            maxmin_error_list = np.mean(maxmin_error_list,axis=0)
            maxmin_error[str(k[i])].append(maxmin_error_list)

            optsim_error_list = np.mean(optsim_error_list,axis=0)
            optisim_error[str(k[i])].append(optsim_error_list)
            
        random_error[str(k[i])] = np.array(random_error[str(k[i])])
        maxmin_error[str(k[i])] = np.array(maxmin_error[str(k[i])])
        optisim_error[str(k[i])] = np.array(optisim_error[str(k[i])])


    for i in range(len(k)):
        random_data = pd.DataFrame(random_error[str(k[i])],columns=['max_error','mean_squared_error','mean_absolute_error'],index=selected_number)
        random_data.to_csv('~/qc_selector/result/sub_sample/random__sample_error_when_k={}.csv'.format(k[i]))

        maxmin_data = pd.DataFrame(maxmin_error[str(k[i])],columns=['max_error','mean_squared_error','mean_absolute_error'],index=selected_number)
        maxmin_data.to_csv('~/qc_selector/result/sub_sample/maxmin__sample_error_when_k={}.csv'.format(k[i]))

        optisim_data = pd.DataFrame(optisim_error[str(k[i])],columns=['max_error','mean_squared_error','mean_absolute_error'],index=selected_number)
        optisim_data.to_csv('~/qc_selector/result/sub_sample/optisim_sample_error_when_k={}.csv'.format(k[i]))
        
