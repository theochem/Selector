#### Intention:

One intention of the evil example we want to explain the question that the error of random sample is better than the diverse sample sometimes. We think when calculating the mean squared error of the biased database, the number of the biased data points is small so that the error will be averaged. When calculating the max error(*the maximum error between two points*), the error of random sample is larger than diverse sample.

---

#### Method:

We construct a biased database with two variables by using the boltzmann sampling on potential energy surface of *Dey, B. K., & Ayers, P. W. (2006). [A Hamilton–Jacobi type equation for computing minimum potential energy paths.](https://www.tandfonline.com/doi/abs/10.1080/00268970500460390) Molecular Physics, 104(4), 541-558.* 

The potential energy surface function of two variables is:

$$
V(q_1,q_2 )=V_0 + a_0\cdot e^{-(q_1-b_{q_1})^2-(q_2-b_{q_2})^2}-\sum_{l=1}^4a_i\cdot e^{-\sigma_i^{q_1}(q_1-\alpha_i)^2-\sigma_{i}^{q_2}(q_2-\beta_i)^2}
$$

Parameters:  

$V_0 = 5.0 \ kcal/mol$ \
$a_0 = 0.6 \ kcal/mol \qquad a_1 = 3.0 \ kcal/mol \qquad a_2 = 1.5 \ kcal/mol \qquad a_3 = 3.2 \ kcal/mol \qquad a_4 = 2.0 \ kcal/mol$ \
$b_{q_1} = 0.1 \ \mathring{A} \qquad b_{q_2} = 0.1 \ \mathring{A}$ \
$\sigma^{q_1} = [0.3,1.0,0.4,1.0]^T \ \mathring{A}^{-2} \qquad \sigma^{q_2} = [0.4,1.0,1.0,0.1]^T \ \mathring{A}^{-2}$ \
$\alpha = [1.3,-1.5,1.4,-1.3]^T \ \mathring{A} \qquad \beta = [-1.6,-1.7,1.8,1.23]^T \ \mathring{A}$ 

Steps:
1. using random sampling to generate 1e6~1e7 points database $D$ (data points from -3 to 3)
2. using the [boltzmann sampling](https://github.com/theochem/DiverseSelector/issues/144#issuecomment-1714408612) with different k on the database $D$ to get the subdatabase $D_s'$
3. using the 20 or more times random sampling to select 1e3 points on the subdatabase $D_s'$ to get the subsubdatabase $D_{ss}''$ 
4. using the random selection, OptiSim and other algorithms to select $S$ points on the subsubdatabase $D_{ss}''$.
5. fitting the PES with Gaussian Process
6. computing mean absolute error, root mean square error, and maximum absolute error for the regular grid of points from -3 to 3 with 1e4 points (each axis may 1e2)
7. average these errors to get smoother plots 
   
Parameter details:

with $\beta=1.5$ by using the $k\cdot \beta$ as the boltzmann parameter (while k euqal 0, the boltzmann sampling is transformed to the random sampling)

---
#### Implementation and Discussion
1. ***evil_example.py*** is the main code.  
   
2. The folder of ***kernel_choose*** include some result of calculating the error with different kernel when using gaussian process.
   * The folder of ***error_data*** include max error and mean squared error of different k (I use random sample sampling 1e3 points on different biased database *(different k)*, the use them to fit the PES by Gaussian process with the different kernel and calculate the error by using the regular grid error(each axis may 1e2 points))
  
   * The folder of ***image*** include some 2D images by fixing the q2 and kernel. (Due to my computer can't plot 3D image, I don't know why and I can't also see the molecular orbitals of Gaussview.)
  
3. The folder of ***biased_unbiased*** include code and images we want to explain that k is lager, the database is more biased. So that when we do random sample on different k database that k is larger, the error is larger(when the selected points number are equal to each other).

4.  The folder of ***different_sample_method*** include image and data with error of different sample method. The result is not good even terrible because of the wrong kernel I mention in Problem part. 
   
5.  The folder of ***points_distribution*** include the points distribution with different k (the initial points is 1e7) 

#### Problem   
The kernel I use before is wrong. So the result in ***different_sample_method*** may be wrong, because I choose the RBF kernel without ConstantKernel. And we can see the fitting of the RBF is terrible. But I think the tendency is right, the oscillation of the random sample maybe origin from the terrible fitting or the nonrobustness of random sample.  kernel.And a strange thing that the curve has a minimum point, I think it maybe derive from the terrible fitting. I'm refitting the PES by using the **(RBF+ConstantKernel)** kernel.

