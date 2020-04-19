# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math
import numpy.random
from random import uniform
from copy import deepcopy
import os
PATH=os.getcwd()
AllSamples = scipy.io.loadmat(PATH+ '//AllSamples.mat')

AllSamples
#AllSamples[0,1]
#randomly pick the initial centers from the given samples
#k can be in range 2-10



# find the distance of first samples from the initial centroid        
def cal_centroid_distance(centroid,data,Clusternum):
    dist=np.zeros((len(data),Clusternum))
    for j in range(Clusternum):
            for i in range(len(data)):
                d_x= math.pow((data[i,0]-centroid[j,0]),2)
                d_y= math.pow((data[i,1]-centroid[j,1]),2)
                d = math.sqrt(d_x + d_y)
                dist[i][j]=d
    return dist         
    
#find the error
def cal_error(c_old,c_new, ClusterNum):
        finalerr=0;
        for i in range(ClusterNum):
            d_x= math.pow((c_new[i,0]-c_old[i,0]),2)
            d_y=math.pow((c_new[i,1]-c_old[i,1]),2)
            err=round(math.sqrt(d_x+d_y),2)
            finalerr=round(err+finalerr,2)
        return finalerr
# Project Part 1
#Strategy 1 - find the initial centers randomly
#create clusters iteratively and update the new centroid till convergence
 
def k_means_random(CNum):
    
    clusters =[0]* CNum 
    distance=np.zeros((len(AllSamples),CNum)) 
    centers=np.zeros((CNum,2))
    centers_old=np.zeros((CNum,2))
    
    for i in range(CNum):
        cluster_x=np.random.random_sample()*10
        cluster_x
        cluster_y=np.random.random_sample()*10
        cluster_y
        centroid_initial= (cluster_x, cluster_y)
        centers[i]=centroid_initial
        diff= cal_error(centers_old,centers, CNum)

# convergence criteria: difference between new and old centroid is zero
    while(diff!=0):
         
          #find the distance of each sample from centroid
         distance=cal_centroid_distance(centers,AllSamples,CNum)
         
         clustermatrix=np.zeros(len(AllSamples))
    
         
     #find the closest sample to cluster
         for i in range(len(AllSamples)):
             mindist=np.argmin(distance[i])
             clustermatrix[i]=mindist
     #find new centroids
         centers_old = deepcopy(centers)
         for i in range(CNum):
             clusterpoints=[AllSamples[j] for j in range(len(AllSamples)) if clustermatrix[j]==i]
             clusters[i]=clusterpoints
             centers[i]=np.mean(clusterpoints, axis=0)
         
         diff= cal_error(centers_old,centers, CNum)
         print(centers)
         
    return centers  
    

 # Calculate the Objective function        
def objective_function(ClusterNum, centr, clusters):
    sumOfClusters =0
    Obj=0
    #print(clusters)
    #for i in range(ClusterNum):
    for arr_i, arr in enumerate(clusters):
        for row_i, row in enumerate(arr):
            d_x=math.pow((row[0]-centr[arr_i,0]),2)
            d_y=math.pow((row[1]-centr[arr_i,1]),2)
            objec=d_x+d_y   
            sumOfClusters=sumOfClusters +objec
    Obj=round(Obj+sumOfClusters,2)
    return Obj

# Change K from 2 to 10 and Plot the graph of clusters against objective functiom
#Strategy 1
k=[2,3,4,5,6,7,8,9,10]
obj=[]
cost=[]
totalcost=0
for i in k:
    NumberofClusters=i
    finalcenter=np.zeros((NumberofClusters,2))
    finalcenter=k_means_random(NumberofClusters)
    print("\n The number of clusters are: ", NumberofClusters)
    print ("\n the centroids of clusters are:\n", finalcenter)
    obj= objective_function(NumberofClusters,finalcenter,clusters)
    cost= np.append(cost,obj)
    totalcost=obj+totalcost
    print("\n the objective function for", NumberofClusters,"is:", obj)
    print("\n the cost is:", cost)
 
    #plot the graph
print("\n the cost is:", cost)
plt.plot(k,cost)
plt.xlabel('number of clusters')
plt.ylabel('objective function')
plt.title('Strategy1')



## Part 2 - Project

#Strategy 2
def k_means_MaxDistSamples(ClusterNum):
    #ClusterNum=2
    clusters_2 =[0]* ClusterNum 
    centers_initial= np.zeros((ClusterNum,2))
    centers_old_2=np.zeros((ClusterNum,2))
    distance_2=np.zeros((len(AllSamples),ClusterNum))
    averageDist=[0]*len(AllSamples)
    # find the first centroid as a random point within the sample
    
    randomInd = np.random.randint(300)
    centers_initial[0]=[AllSamples[randomInd][0],AllSamples[randomInd][1]]
   # find the next centroid which is farthest from first centroid in the data set
    distance_2=cal_centroid_distance(centers_initial,AllSamples,1)
         
    maxdist=np.argmax(distance_2)
    centers_initial[1]= [AllSamples[maxdist][0], AllSamples[maxdist][1]]
    
    if ClusterNum>2:
    #calculate the distance of first two centroids from all the samples, say d1 and d2 
    #then take the average of these distances and then choose the farthest average as the third centroid
        for i in range(2,ClusterNum,1):
            distance_2= cal_centroid_distance(centers_initial, AllSamples,i)
            for j in range(len(distance_2)):
                averageDist[j]= round(np.mean(distance_2[j]),2)
            
            maxdist=np.argmax(averageDist)
            centers_initial[i]= [AllSamples[maxdist][0], AllSamples[maxdist][1]]
    
    diff2= cal_error(centers_old_2,centers_initial, ClusterNum)

    while(diff2!=0):
         
          #find the distance of each sample from centroid
         distance_2=cal_centroid_distance(centers_initial,AllSamples,ClusterNum)
         
         clustermatrix2=np.zeros(len(AllSamples))
    
         
     #find the closest sample to cluster
         for i in range(len(AllSamples)):
             mindist=np.argmin(distance_2[i])
             clustermatrix2[i]=mindist
     #find new centroids
         centers_old_2 = deepcopy(centers_initial)
         for i in range(ClusterNum):
             clusterpoints=[AllSamples[j] for j in range(len(AllSamples)) if clustermatrix2[j]==i]
             clusters_2[i]=clusterpoints
             centers_initial[i]=np.mean(clusterpoints, axis=0)
    
         mask = np.isnan(centers_initial).any(axis=1)
         centers_initial[mask]= np.mean(centers_initial[~mask])
         diff2= cal_error(centers_old_2,centers_initial, ClusterNum)
         print(centers_initial)
   
    return centers_initial

#Strategy 2- Objective function     
def objective_function_2(ClusterNum, centr,clusters_2):
    sumOfClusters2 =0
    ObjectiveSum2 =0
    #print(clusters)
    #for i in range(ClusterNum):
    for arr_i, arr in enumerate(clusters_2):
        for row_i, row in enumerate(arr):
            d_x2=math.pow((row[0]-centr[arr_i,0]),2)
            d_y2=math.pow((row[1]-centr[arr_i,1]),2)
            objec2=d_x2+d_y2  
            sumOfClusters2=sumOfClusters2 +objec2
    ObjectiveSum2= round(sumOfClusters2 + ObjectiveSum2,2)
        
    return ObjectiveSum2  



    
    #Strategy 2
k=[2,3,4,5,6,7,8,9,10]
obj2=[]
cost2=[]
totalcost2=0
for i in k:
    NumofClusters=i
    finalcenter2=np.zeros((NumofClusters,2))
    finalcenter2 =k_means_MaxDistSamples(NumofClusters)
    print("\n The number of clusters are: ", NumofClusters)
    print ("\n the centroids of clusters are:\n", finalcenter2)
    obj2= objective_function_2(NumofClusters,finalcenter2, clusters_2)
   # obj2= objective_function_2(NumofClusters,centers_initial, clusters_2)
    totalcost2=totalcost2+obj2
    cost2= np.append(cost2,obj2)
    print("\n the objective function for clusters," ,NumofClusters , "is:", obj2)
    print("\n the cost is:", cost2)
    
    

#print("\n The final objective cost K from 2 to 10 is:", totalcost2)
print("\n the cost is:", cost2)
plt.plot(k,cost2)
plt.xlabel('number of clusters')
plt.ylabel('objective function')
plt.title('Strategy2')
    
      
     
#         colors=['r', 'm','g','y','c','k']
#         fig, gp =plt.subplots()
#         for i in range(3):
#             clusterpoints=np.array([AllSamples[j] for j in range(len(AllSamples)) if clustermatrix[j]==i])
#             gp.scatter(clusterpoints[:,0], clusterpoints[:,1], c=colors[i])
#         gp.scatter(centers[:,0], centers[:,1], marker='+', c='b') 
    #compare with the objective function
    
#plot the graphs 

