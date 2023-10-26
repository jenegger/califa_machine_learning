'''
Collection of some generating classes for some "toy problems",
although maybe we can extend to the real data also.

Nicole Hartman
Summer 2023
'''
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import math
from numpy import genfromtxt

def make_batch(
    device='cpu',
    N_events=10,
    N_clusters=3,
    isRing=True,
    blurImage=False
):
    my_data = genfromtxt('/home/tobias/califa_ML_model/code/data_stream_2121.txt', delimiter=',')
    my_data[:,4] = my_data[:,4]+4500
    array_unique_theta = np.unique(my_data[:,2])
    array_unique_phi = np.unique(my_data[:,3])
    
    for i in range(len(array_unique_theta)):
        array_unique_theta[i] = math.trunc(array_unique_theta[i])
    for i in range(len(array_unique_phi)):
        array_unique_phi[i] = math.trunc(array_unique_phi[i])
    array_unique_theta = np.unique(array_unique_theta)
    array_unique_phi = np.unique(array_unique_phi)
    array_unique_events = np.unique(my_data[:,0])
    selected_hits = np.empty(0)
    selected_hits = np.random.choice(array_unique_events,N_events*N_clusters,replace=False)
    selected_hits = np.resize(selected_hits,(N_events,N_clusters))
    
    
    
    evt_histogram_array = np.zeros((N_events,27,112))
    
    for i in range(N_events): 
        for j in range(N_clusters):
            hitnr = selected_hits[i][j]
            hit = my_data[my_data[:,0] == hitnr]
            for k in range(hit.shape[0]):
                pixel_x = np.argwhere(array_unique_theta == math.trunc(hit[k][2]))[0]
                pixel_y = np.argwhere(array_unique_phi == math.trunc(hit[k][3]))[0]
                e_val = hit[k][1]
                evt_histogram_array[i,pixel_x,pixel_y] = e_val
    #with np.printoptions(threshold=np.inf):
    #    print(evt_histogram_array[9,:,:])
    summed_evt_histogram_array = np.sum(evt_histogram_array,axis=0)
    #with np.printoptions(threshold=np.inf):
    #    print(summed_evt_histogram_array)
    
    
    evt_masks = np.zeros((N_events,N_clusters,27,112))
    for i in range(N_events):
        for j in range(N_clusters):
            hitnr = selected_hits[i][j]
            hit = my_data[my_data[:,0] == hitnr]
            for k in range(hit.shape[0]):
                pixel_x = np.argwhere(array_unique_theta == math.trunc(hit[k][2]))[0]
                pixel_y = np.argwhere(array_unique_phi == math.trunc(hit[k][3]))[0]
                e_val = hit[k][1]
                if summed_evt_histogram_array[pixel_x,pixel_y] > 0:
                    evt_masks[i,j,pixel_x,pixel_y] = e_val/summed_evt_histogram_array[pixel_x,pixel_y]
    
    evt_histogram_array = evt_histogram_array[:,None,:,:] 
    return torch.FloatTensor(evt_histogram_array).to(device), \
            torch.FloatTensor(evt_masks).to(device)



def gen_events(
    bins,
    N_events = 100000,
    N_clusters = 1,
    #lam_n_clusters = 1.0,
    #lam_N_events_in_cluster = 200.,
    lam_noise = 0., #0.1
    mean_var_cluster = np.log(0.001),
    sigma_var_cluster = 0.1,
    mean_lam_cluster = np.log(200.),
    sigma_lam_cluster = 1.,
    isRing=True,
    mirrorSamples=False,
    blurImage=False,
    xlow=-.5,
    xhigh=0.5,
    stdlow=0.01,
    stdhigh=0.05
):
    '''
    Code from Florian
    https://gitlab.lrz.de/neural_network_clustering/permutation_invariant_loss/-/blob/main/test_blur.ipynb
    '''

    eventHistograms = np.zeros(shape=(N_events, len(bins)-1, len(bins)-1,1) ) #, dtype=np.float32)
    eventNumbers = np.zeros(N_events)

    nMaxClusters = N_clusters   
    
    eventInfo = np.zeros(shape=(N_events, nMaxClusters, 4)) #, dtype=np.float32)
    for iEvent in range(N_events):
        
        image = np.zeros_like(eventHistograms[iEvent,:,:,0])
        
        n_clusters = nMaxClusters

        eventNumbers[iEvent] += n_clusters
        eI = []
        for iCluster in range(min(n_clusters,nMaxClusters)):
            # how many events in this cluster

            lam_N_events_in_cluster = 200. #min(np.random.lognormal(mean=mean_lam_cluster, sigma=sigma_lam_cluster),400)

            N_events_in_cluster = np.random.poisson(lam_N_events_in_cluster)

            # where is the cluster center
            cluster_center = np.random.uniform(low=xlow, high=xhigh, size=2)

            # what is the cluster spread
            var_cluster = np.random.uniform(stdlow,stdhigh) #np.random.lognormal(mean=mean_var_cluster, sigma=sigma_var_cluster)

            cluster_events_x0 = np.random.normal(loc=0., scale=1., size=N_events_in_cluster)
            cluster_events_y0 = np.random.normal(loc=0., scale=1., size=N_events_in_cluster)

            if isRing:
                fact = np.sqrt(var_cluster/(cluster_events_x0**2+cluster_events_y0**2))
            else:
                fact = np.sqrt(var_cluster)

            cluster_events_x = cluster_events_x0*fact + cluster_center[0]
            cluster_events_y = cluster_events_y0*fact + cluster_center[1]
            
            # bin the events
            H, _, _ = np.histogram2d(cluster_events_x, cluster_events_y, bins=[bins,bins])

            image += H.T
            
            blur = np.random.uniform(0, 5.)*blurImage
            image =  gaussian_filter(image,sigma=blur)
            
            eI.append(np.concatenate([cluster_center, [var_cluster,blur]]))

        eventHistograms[iEvent,:,:,0] = np.copy(image)
            
        if lam_noise > 0:
            eventHistograms[iEvent] += np.random.poisson(lam_noise, size=(len(bins)-1)**2).reshape((len(bins)-1, len(bins)-1))
        #eventHistograms[iEvent] /= np.sum(eventHistograms[iEvent])
        #norm = np.sum(eventHistograms[iEvent])
        #if norm == 0:
        #    norm = 1.
        #if np.sum(eventHistograms[iEvent]) == 0:
        #    print("help!!")
        #eventHistograms[iEvent] /= norm
 
        eventInfo[iEvent] = np.array(eI)
    
    if mirrorSamples:
        
        eventHistograms = np.concatenate([eventHistograms, eventHistograms[:,:,::-1], eventHistograms[:,::-1,:],eventHistograms[:,::-1,::-1]] )
        
        eventInfoFlippedX = np.copy(eventInfo)
        eventInfoFlippedX[:,:,0] *= -1
        
        eventInfoFlippedY = np.copy(eventInfo)
        eventInfoFlippedY[:,:,1] *= -1
        
        eventInfoFlippedXY = np.copy(eventInfo)
        eventInfoFlippedXY[:,:,0] *= -1
        eventInfoFlippedXY[:,:,1] *= -1
        
        eventInfo = np.concatenate([eventInfo, eventInfoFlippedX, eventInfoFlippedY, eventInfoFlippedXY])
    
    return eventInfo.astype('float32'), eventHistograms.astype('float32')


class ToyProblem():
    
    def __init__():
        '''
        '''
        pass
    
    def generateData(nExamples, nPhotons, resolution,range_gen=(-2,2)):
                 
        '''
        Inputs: 
        - nExamples: # of events to draw (for training and test)
        - nPhotons: object cardinality in the event
        - resolution:
        - range_gen: Box where the center of the photons can be
        '''

        # Setup
        nCoords = 2 # (x,y) and the E


        '''
        Step 1: draw positions for the photon coordinate
        `C`:
        - C[i,0] is the (x,y) location of the first "photon" in event i 
        - C[i,1] is the (x,y) location of the second "photon" in event i

        Consider photon energies uniformly distributed from 5 -- 500 GeV
        '''
        C = np.random.uniform(*range_gen,size=(nExamples,nPhotons,1,nCoords)) # coords
        E = np.random.uniform(5,500,size=(nExamples, nPhotons,1)) # energies

        Y = np.concatenate([C.squeeze(),E],axis=-1)

        '''
        Step 2: Simulate the calorimeter images for each of these photon clusters
        '''
        X_photons   = [[] for i in range (nPhotons) ]
        stop_viz = 16

        # Calorimeter images, energy deposited in each cell
        imgs = np.zeros((nExamples, *resolution, 1)) # Shape (nExamples, 9, 9, 1) 

        for i, (Es) in enumerate(E.squeeze()):

            Xi = []

            for j, E_photon in enumerate(Es): 

                # Very simple model, lets assume there's 1 photon produced per GeV of energy.
                nSamples=int(E_photon)
                x_j = C[i,j] + np.random.randn(nSamples, nCoords)

                Xi.append(x_j) # Append to get the calo img for this event

                if i < stop_viz:
                    X_photons[j].append(x_j.squeeze()) # Append to make scatter plots

            # Concatenate the calorimeter image from all of these photons
            Xi = np.concatenate(Xi, axis=0)

            # Get the histgram image
            imgs[i] = np.histogram2d(*Xi.T, resolution, [(-4.5,4.5),(-4.5,4.5)])[0].reshape(1,*resolution,1) 

        return imgs, Y, X_photons
