#!/usr/bin/env python
# coding: utf-8

# # Getting started
# ## Demo - Dissimilarity Based Selection Synthetic Data

# In this tutorial, we are going to show how the `DiverseSelector` work with 2D synthstic data. Each axis represents one feature. For each example, we will have two different synthstic datasets where the first one can be seen as randomly generated data points and the latter generates data points beloing to different clusters. By using 2D feature data points, it makes visualization easy.

# ## Generating Synthetic Data

# In[ ]:


# install the DiverseSelector module
get_ipython().system('pip install git+https://github.com/theochem/DiverseSelector')


# In[1]:


import DiverseSelector
from DiverseSelector.test.common import generate_synthetic_data
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


coords, class_labels, arr_dist = generate_synthetic_data(n_samples=100,
                                                         n_features=2,
                                                         n_clusters=1,
                                                         pairwise_dist=True,
                                                         metric="euclidean",
                                                         random_state=42)

coords_cluster, class_labels_cluster, arr_dist_cluster = generate_synthetic_data(n_samples=100,
                                                         n_features=2,
                                                         n_clusters=3,
                                                         pairwise_dist=True,
                                                         metric="euclidean",
                                                         random_state=42)
def graph_data(coords, selected=None, reference=False):
    plt.figure(dpi=150)
    plt.scatter(coords[:, 0], coords[:, 1])
    if selected:
        for i, mol_id in enumerate(selected):
            plt.scatter(coords[mol_id, 0], coords[mol_id, 1], c='r')
            plt.text(coords[mol_id,0], coords[mol_id,1], str(i+1))
    if reference:
        plt.scatter(coords[0, 0], coords[0, 1], c='black')
    plt.show()


# In[3]:


graph_data(coords)
graph_data(coords_cluster)


# ## Running Dissimilarity Algorithms

# ### Brute Strength - MaxMin

# In[8]:


# MaxMin method
from DiverseSelector.selectors import MaxMin, OptiSim, DirectedSphereExclusion, GridPartitioning

# diverse subset selection from randomly generated data points
selector = MaxMin()
selected_ids1 = selector.select(arr=arr_dist, num_selected=12)

# diverse subset selection from data points with obvious patterns (different clusters)
selector = MaxMin()
selected_ids2 = selector.select(arr=arr_dist_cluster, labels=class_labels_cluster, num_selected=12)


# In[9]:


graph_data(coords,selected_ids1)
graph_data(coords_cluster,selected_ids2)


# ### Adapted Optimizable K-Dissimilarity Selection (OptiSim)

# In[12]:


# diverse subset selection from randomly generated data points
selected_id3 = OptiSim().select(coords, 12)
# diverse subset selection from data points with obvious patterns (different clusters)
selected_id4 = OptiSim().select(coords_cluster, num_selected=12, labels=class_labels_cluster)

graph_data(coords,selected_id3)
graph_data(coords_cluster,selected_id4)


# ### Directed Sphere Exclusion

# In[13]:


# diverse subset selection from randomly generated data points
selected_id5 = DirectedSphereExclusion().select(coords, 12)
# diverse subset selection from data points with obvious patterns (different clusters)
selected_id6 = DirectedSphereExclusion().select(coords_cluster, num_selected=12, labels=class_labels_cluster)

graph_data(coords,selected_id5)
graph_data(coords_cluster,selected_id6)


# ### Grid Partitioning Method

# In[15]:


# diverse subset selection from randomly generated data points
selected_id7 = GridPartitioning(2, "equisized_independent").select(coords, 12)
# diverse subset selection from data points with obvious patterns (different clusters)
selected_id8 = GridPartitioning(2, "equisized_independent").select(coords_cluster, 
                                                                   num_selected=12, 
                                                                   labels=class_labels_cluster)

graph_data(coords,selected_id7)
graph_data(coords_cluster,selected_id8)
# 20, 5, 3, "equisized_independent"


# ## MinMax selection with clusters, but without assuming there are clusters

# In[16]:


selector = MaxMin()
selected_ids = selector.select(arr=arr_dist_cluster, num_selected=15)
graph_data(coords_cluster, selected_ids)


# In[12]:


import numpy as np
from sklearn.metrics import pairwise_distances
np.random.seed(42)
cluster_one = np.random.normal(0, 1, (3,2))
cluster_two = np.random.normal(10, 1, (6,2))
cluster_three = np.random.normal(20, 1, (7,2))
labels_mocked = np.hstack([[0 for i in range(3)],
                          [1 for i in range(6)],
                          [2 for i in range(7)]])

mocked_cluster_coords = np.vstack([cluster_one, cluster_two, cluster_three])
selector = MaxMin(lambda x: pairwise_distances(x, metric='euclidean'))
selected_mocked = selector.select(mocked_cluster_coords, num_selected=15, labels=labels_mocked)

graph_data(mocked_cluster_coords, selected_mocked)

selected_mocked


# ### Brute Strength - maxsum

# In[6]:


selector = DissimilaritySelection(num_selected=12,
                                  arr_dist=arr_dist,
                                  random_seed=42,
                                  method="maxsum")
selector.starting_idx = 0
selected_ids2 = selector.select()


# In[7]:


graph_data(selected_ids2)


# ### Grid Partitioning - Equisized Independent

# In[8]:


selector = DissimilaritySelection(features=coords,
                                  num_selected=12,
                                  arr_dist=arr_dist,
                                  grid_method="grid_partioning",
                                  random_seed=42)
selector.starting_idx = 0
selected_ids3 = selector.select()


# In[9]:


graph_data(selected_ids3)


# ### Grid Partitioning - Equisized Dependent

# In[10]:


selector = DissimilaritySelection(num_selected=12,
                                  features = coords,
                                  random_seed=42,
                                  dissim_func="grid_partitioning",
                                  grid_method="equisized_dependent")
selector.starting_idx = 0
selected_ids4 = selector.select()


# In[11]:


graph_data(selected_ids4)


# ### Directed Sphere Exclusion

# In[12]:


selector = DissimilaritySelection(num_selected=12,
                                  features=coords,
                                  dissim_func="sphere_exclusion",
                                  random_seed=42)
selector.starting_idx = 0
selected_ids5 = selector.select()


# In[13]:


graph_data(selected_ids5, True)


# ### Optisim

# In[14]:


selector = DissimilaritySelection(num_selected=12,
                                  features=coords,
                                  dissim_func="optisim",
                                  random_seed=42)
selector.starting_idx = 0
selected_ids6 = selector.select()


# In[15]:


graph_data(selected_ids6)

