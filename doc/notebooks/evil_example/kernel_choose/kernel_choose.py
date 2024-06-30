import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF,Exponentiation,RationalQuadratic,ExpSineSquared,DotProduct
from sklearn.metrics import mean_absolute_error,mean_squared_error,pairwise_distances,max_error


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


def generate_data2(number):
    q1 = np.linspace(-3,3,number)    

    data = []
    for i in range(len(q1)):
        data.append([q1[i],1.8])

    data = np.array(data)
    
    return data


if __name__ == '__main__':
    # generate a total random database
    rng = np.random.default_rng(seed=42)
    total_database = (rng.random((int(1e7),2))-0.5)*6

    # using boltzmann sampling to get a sub_database
    k = 2
    sub_database = boltzmann_sample(total_database,k)

    sub_sub_database = random_sample(sub_database,1000,1)
    sub_sub_database = sub_sub_database[0,:]

    X_train = sub_sub_database
    Y_train = potential_energy(sub_sub_database[:,0],sub_sub_database[:,1])

    name_list = []
    max_error_list = []
    mean_error_list=[]


    # kernel = DP
    kernel = DotProduct()
    gpr = GaussianProcessRegressor(n_restarts_optimizer=20,kernel=kernel)
    gpr = gpr.fit(X_train,Y_train)

    # calculate total error
    X_test = generate_data(100)
    Y_test = gpr.predict(X_test)
    Y_ture = potential_energy(X_test[:,0],X_test[:,1])

    name_list.append('DP')
    max_error_list.append(max_error(Y_test,Y_ture))
    mean_error_list.append(mean_squared_error(Y_test,Y_ture))

    # plot
    # X_test = generate_data2(1000)
    # Y_test,sigma = gpr.predict(X_test,return_std=True)

    # plt.title('kernel : DotProduct')
    # plt.xlabel('q1')
    # plt.ylabel('potential_energy(kcal/mol)')
    # plt.plot(X_test[:,0],potential_energy(X_test[:,0],X_test[:,1]),label='true curve')
    # plt.plot(X_test[:,0],Y_test,label='fit curve')
    # plt.fill_between(X_test[:,0],Y_test-1.96*sigma,Y_test+1.96*sigma,alpha=0.1,color='r'
    #                  ,label=u'95% confidence interval')
    # plt.legend()


    # kernel = DP + C
    kernel = DotProduct() + ConstantKernel(constant_value=2)
    gpr = GaussianProcessRegressor(n_restarts_optimizer=20,kernel=kernel)
    gpr = gpr.fit(X_train,Y_train)

    # calculate total error
    X_test = generate_data(100)
    Y_test = gpr.predict(X_test)
    Y_ture = potential_energy(X_test[:,0],X_test[:,1])

    name_list.append('DP+C')
    max_error_list.append(max_error(Y_test,Y_ture))
    mean_error_list.append(mean_squared_error(Y_test,Y_ture))

    # plot
    # X_test = generate_data2(1000)
    # Y_test,sigma = gpr.predict(X_test,return_std=True)

    # plt.title('kernel : DotProduct and ConstantKernel')
    # plt.xlabel('q1')
    # plt.ylabel('potential_energy(kcal/mol)')
    # plt.plot(X_test[:,0],potential_energy(X_test[:,0],X_test[:,1]),label='true curve')
    # plt.plot(X_test[:,0],Y_test,label='fit curve')
    # plt.fill_between(X_test[:,0],Y_test-1.96*sigma,Y_test+1.96*sigma,alpha=0.1,color='r'
    #                  ,label=u'95% confidence interval')
    # plt.legend()


    # kernel = RQ
    kernel = RationalQuadratic(length_scale=24,length_scale_bounds=(1e-5,1e5), alpha=1)
    gpr = GaussianProcessRegressor(n_restarts_optimizer=20,kernel=kernel)
    gpr = gpr.fit(X_train,Y_train)

    # calculate total error
    X_test = generate_data(100)
    Y_test = gpr.predict(X_test)
    Y_ture = potential_energy(X_test[:,0],X_test[:,1])

    name_list.append('RQ')
    max_error_list.append(max_error(Y_test,Y_ture))
    mean_error_list.append(mean_squared_error(Y_test,Y_ture))

    # plot
    # X_test = generate_data2(1000)
    # Y_test,sigma = gpr.predict(X_test,return_std=True)

    # plt.title('kernel : RationQuadratic')
    # plt.xlabel('q1')
    # plt.ylabel('potential_energy(kcal/mol)')
    # plt.plot(X_test[:,0],potential_energy(X_test[:,0],X_test[:,1]),label='true curve')
    # plt.plot(X_test[:,0],Y_test,label='fit curve')
    # plt.fill_between(X_test[:,0],Y_test-1.96*sigma,Y_test+1.96*sigma,alpha=0.1,color='r'
    #                  ,label=u'95% confidence interval')
    # plt.legend()


    # kernel = RQ + C
    kernel = RationalQuadratic(length_scale=24,alpha=1) + ConstantKernel(constant_value=2)
    gpr = GaussianProcessRegressor(n_restarts_optimizer=20,kernel=kernel)
    gpr = gpr.fit(X_train,Y_train)

    # calculate total error
    X_test = generate_data(100)
    Y_test = gpr.predict(X_test)
    Y_ture = potential_energy(X_test[:,0],X_test[:,1])

    name_list.append('RQ+C')
    max_error_list.append(max_error(Y_test,Y_ture))
    mean_error_list.append(mean_squared_error(Y_test,Y_ture))

    # plot
    # X_test = generate_data2(1000)
    # Y_test,sigma = gpr.predict(X_test,return_std=True)

    # plt.title('kernel : RationQuadratic and ConstantKernel')
    # plt.xlabel('q1')
    # plt.ylabel('potential_energy(kcal/mol)')
    # plt.plot(X_test[:,0],potential_energy(X_test[:,0],X_test[:,1]),label='true curve')
    # plt.plot(X_test[:,0],Y_test,label='fit curve')
    # plt.fill_between(X_test[:,0],Y_test-1.96*sigma,Y_test+1.96*sigma,alpha=0.1,color='r'
    #                  ,label=u'95% confidence interval')
    # plt.legend()


    # kernel = RBF
    kernel = RBF(length_scale=1, length_scale_bounds=(1e-7, 1e7))
    gpr = GaussianProcessRegressor(n_restarts_optimizer=20,kernel=kernel)
    gpr = gpr.fit(X_train,Y_train)

    # calculate total error
    X_test = generate_data(100)
    Y_test = gpr.predict(X_test)
    Y_ture = potential_energy(X_test[:,0],X_test[:,1])

    name_list.append('RBF')
    max_error_list.append(max_error(Y_test,Y_ture))
    mean_error_list.append(mean_squared_error(Y_test,Y_ture))

    # plot
    # X_test = generate_data2(1000)
    # Y_test,sigma = gpr.predict(X_test,return_std=True)
    # plt.title('kernel : RBF')
    # plt.xlabel('q1')
    # plt.ylabel('potential_energy(kcal/mol)')
    # plt.plot(X_test[:,0],potential_energy(X_test[:,0],X_test[:,1]),label='true curve')
    # plt.plot(X_test[:,0],Y_test,label='fit curve')
    # plt.fill_between(X_test[:,0],Y_test-1.96*sigma,Y_test+1.96*sigma,alpha=0.1,color='r'
    #                  ,label=u'95% confidence interval')

    # plt.legend()


    # kernel = RBF + C
    kernel = RBF(length_scale=1, length_scale_bounds=(1e-7, 1e7)) + ConstantKernel(constant_value=2)
    gpr = GaussianProcessRegressor(n_restarts_optimizer=20,kernel=kernel)
    gpr = gpr.fit(X_train,Y_train)

    # calculate total error
    X_test = generate_data(100)
    Y_test = gpr.predict(X_test)
    Y_ture = potential_energy(X_test[:,0],X_test[:,1])

    name_list.append('RBF+C')
    max_error_list.append(max_error(Y_test,Y_ture))
    mean_error_list.append(mean_squared_error(Y_test,Y_ture))

    # plot
    # X_test = generate_data2(1000)
    # Y_test,sigma = gpr.predict(X_test,return_std=True)
    # plt.title('kernel : RBF and ConstantKernel')
    # plt.xlabel('q1')
    # plt.ylabel('potential_energy(kcal/mol)')
    # plt.plot(X_test[:,0],potential_energy(X_test[:,0],X_test[:,1]),label='true curve')
    # plt.plot(X_test[:,0],Y_test,label='fit curve')
    # plt.fill_between(X_test[:,0],Y_test-1.96*sigma,Y_test+1.96*sigma,alpha=0.1,color='r'
    #                  ,label=u'95% confidence interval')

    # plt.legend()


    # kernel = RBF + DP + C
    kernel = RBF(length_scale=1, length_scale_bounds=(1e-7, 1e7)) + DotProduct() + ConstantKernel(constant_value=2)
    gpr = GaussianProcessRegressor(n_restarts_optimizer=20,kernel=kernel)
    gpr = gpr.fit(X_train,Y_train)

    # calculate total error
    X_test = generate_data(100)
    Y_test = gpr.predict(X_test)
    Y_ture = potential_energy(X_test[:,0],X_test[:,1])

    name_list.append('RBF+DP+C')
    max_error_list.append(max_error(Y_test,Y_ture))
    mean_error_list.append(mean_squared_error(Y_test,Y_ture))

    # plot
    # X_test = generate_data2(1000)
    # Y_test,sigma = gpr.predict(X_test,return_std=True)
    # plt.title('kernel : RBF and DotProduct and ConstantKernel')
    # plt.xlabel('q1')
    # plt.ylabel('potential_energy(kcal/mol)')
    # plt.plot(X_test[:,0],potential_energy(X_test[:,0],X_test[:,1]),label='true curve')
    # plt.plot(X_test[:,0],Y_test,label='fit curve')
    # plt.fill_between(X_test[:,0],Y_test-1.96*sigma,Y_test+1.96*sigma,alpha=0.1,color='r'
    #                  ,label=u'95% confidence interval')

    # plt.legend()


    # kernel = ExpSin
    kernel = ExpSineSquared(length_scale=24, periodicity=1)
    gpr = GaussianProcessRegressor(n_restarts_optimizer=20,kernel=kernel)
    gpr = gpr.fit(X_train,Y_train)

    # calculate total error
    X_test = generate_data(100)
    Y_test = gpr.predict(X_test)
    Y_ture = potential_energy(X_test[:,0],X_test[:,1])

    name_list.append('ExpSin')
    max_error_list.append(max_error(Y_test,Y_ture))
    mean_error_list.append(mean_squared_error(Y_test,Y_ture))

    # plot
    # X_test = generate_data2(1000)
    # Y_test,sigma = gpr.predict(X_test,return_std=True)

    # plt.title('kernel : ExpSineSquared')
    # plt.xlabel('q1')
    # plt.ylabel('potential_energy(kcal/mol)')
    # plt.plot(X_test[:,0],potential_energy(X_test[:,0],X_test[:,1]),label='true curve')
    # plt.plot(X_test[:,0],Y_test,label='fit curve')
    # plt.fill_between(X_test[:,0],Y_test-1.96*sigma,Y_test+1.96*sigma,alpha=0.1,color='r'
    #                  ,label=u'95% confidence interval')
    # plt.legend()


    # kernel = ExpSin + C
    kernel = ExpSineSquared(length_scale=24, periodicity=1) + ConstantKernel(constant_value=2)
    gpr = GaussianProcessRegressor(n_restarts_optimizer=20,kernel=kernel)
    gpr = gpr.fit(X_train,Y_train)

    # calculate total error
    X_test = generate_data(100)
    Y_test = gpr.predict(X_test)
    Y_ture = potential_energy(X_test[:,0],X_test[:,1])

    name_list.append('ExpSin+C')
    max_error_list.append(max_error(Y_test,Y_ture))
    mean_error_list.append(mean_squared_error(Y_test,Y_ture))

    # plot
    # X_test = generate_data2(1000)
    # Y_test,sigma = gpr.predict(X_test,return_std=True)

    # plt.title('kernel : ExpSineSquared + ConstantKernel')
    # plt.xlabel('q1')
    # plt.ylabel('potential_energy(kcal/mol)')
    # plt.plot(X_test[:,0],potential_energy(X_test[:,0],X_test[:,1]),label='true curve')
    # plt.plot(X_test[:,0],Y_test,label='fit curve')
    # plt.fill_between(X_test[:,0],Y_test-1.96*sigma,Y_test+1.96*sigma,alpha=0.1,color='r'
    #                  ,label=u'95% confidence interval')
    # plt.legend()


    # kernel = RBF + RQ + ExpSin + C
    kernel = RBF(length_scale=1, length_scale_bounds=(1e-7, 1e7)) + ConstantKernel(constant_value=2) + RationalQuadratic(length_scale=24, alpha=0.5,
                                                    length_scale_bounds=(1e-5,1e5),alpha_bounds=(1e-2,1e2))+ExpSineSquared(length_scale=24,periodicity=1)
    gpr = GaussianProcessRegressor(n_restarts_optimizer=20,kernel=kernel)
    gpr = gpr.fit(X_train,Y_train)

    # calculate total error
    X_test = generate_data(100)
    Y_test = gpr.predict(X_test)
    Y_ture = potential_energy(X_test[:,0],X_test[:,1])

    name_list.append('RBF+RQ+ExpSin+C')
    max_error_list.append(max_error(Y_test,Y_ture))
    mean_error_list.append(mean_squared_error(Y_test,Y_ture))

    # plot
    # X_test = generate_data2(1000)
    # Y_test, sigma = gpr.predict(X_test,return_std=True)

    # plt.title('kernel : RBF + ExpSineSquared + RationQuadratic + ConstantKernel')
    # plt.xlabel('q1')
    # plt.ylabel('potential_energy(kcal/mol)')
    # plt.plot(X_test[:,0],potential_energy(X_test[:,0],X_test[:,1]),label='true curve')
    # plt.plot(X_test[:,0],Y_test,label='fit curve')
    # plt.fill_between(X_test[:,0],Y_test-1.96*sigma,Y_test+1.96*sigma,alpha=0.1,color='r'
    #                  ,label=u'95% confidence interval')
    # plt.legend()

    # create a csv file
    data = {'max_error':max_error_list,'mean_squared_error':mean_error_list}
    data = pd.DataFrame(data,index=name_list)
    data.to_csv('error when k=2.csv')