#%% import packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
import statistics
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from scipy import interpolate
import matplotlib as mpl
import pickle

#%% Import HDF5 file and generate a 3D array of sweeps, timepoints, wavelengths
def import_hdf5(file):                           
    with h5py.File(file,'r') as f:
        sweep_shape = f['Sweeps']['Sweep_0'].shape            # Shape of each sweep (constant between sweeps) - timepoints x wavelengths
        num_sweeps = len(f['Sweeps'])                         # Number of sweeps (length of sweeps file)
        sweep_data = np.zeros((num_sweeps, *sweep_shape))     # Creates array of zeros of correct shape of total array
        for i, sweep_name in enumerate(f['Sweeps']):          # Populates array of zeros with data to create 
            sweep_data[i] = f['Sweeps'][sweep_name][()]
    return sweep_data

#%% Mean of sweeps
def mean_sweeps(arr3d):
    if arr3d.ndim != 3:                                        # Makes sure input is a 3d array
        raise ValueError("Input array must be 3-dimensional")
    arr2d = np.mean(arr3d, axis=0, dtype=arr3d.dtype)          # Mean along 0th axis (sweeps) to give average signal
    
    return arr2d

#%% Remove Background
def rem_bg(arr2d, threshold=-2, exclude_rows=1, exclude_cols=1, axis=0, plot=False):
    if arr2d.ndim != 2 or arr2d.shape[axis] <= exclude_cols:                        # Specify 2d array/ensure enough columns for mean
        raise ValueError("Input array has invalid shape or dimensions - needs 2")
    
    bkg_time = np.where(arr2d[1:,0] <= threshold)[0]                                # start to threshold value as time to calcualte bkg  
    if len(bkg_time) == 0:
        raise ValueError("No sampletime indices found below threshold")             # Ensure you are sampling something
    
    bg = np.mean(arr2d[bkg_time+1, exclude_cols:], axis=axis)                         # Calculate background
    arr2d[exclude_rows:, exclude_cols:] = arr2d[exclude_rows:, exclude_cols:] - bg[None, :] # Sub background from rest of array
    
    if plot == True:
        plt.plot(arr2d[0,1:], bg)
        plt.x_label('Wavelength (nm)')
        plt.y_label('Background')
    return arr2d


#%% Heat Map
def heatmap(arr2d, wlmin = 100, wlmax = -10, maxTime = None, tUnit = 's',
            x = 1000.0, title = None, cScale = 'viridis', rasterized = True):
    if arr2d.ndim != 2:
        raise ValueError("Input array must be 2D") # Make sure input is 2D array

    if maxTime is None:                            # If no maxTime is specified, use the whole array
        maxTime = arr2d.shape[0]

    yaxis_labels = np.around(arr2d[0, wlmin:wlmax].flatten()) # Create y-axis labels

    xaxis_labels = np.around(arr2d[1:maxTime, 0]) # Create x-axis labels
    d = x * arr2d[1:maxTime, wlmin:wlmax].T # Create data array
    
    fig, ax = plt.subplots()
    a = ax.pcolormesh(xaxis_labels, yaxis_labels, d, cmap=cScale, linewidth=0, rasterized=rasterized,shading = 'gouraud',vmin=d.min(),vmax=d.max(),levels=(np.linspace(d.min(),d.max(),1000))) # Create heatmap
    A = int(np.log10(abs(1 / x)))
    cbar = fig.colorbar(a, ax=ax) # Create colourbar
    cbar.set_label(f'${{\Delta}}$T/T x 10$^{{{A}}}$') # Set colourbar label
    ax.set_ylabel('Wavelength (nm)')
    ax.set_xlabel(f'Time ({tUnit})')
    ax.set_xscale('symlog',linthresh=2)
    ax.set_xlim(xaxis_labels[0], xaxis_labels[-1])

        
    formatter = ticker.ScalarFormatter(useMathText=True) # create a ScalarFormatter object
    formatter.set_scientific(False) # set the label format to plain (i.e., non-scientific notation)
    ax.yaxis.set_major_formatter(formatter) # set the y-axis tick labels to use the ScalarFormatter object

    ax.set_yticks(yaxis_labels[::10].astype(int)) # Set y-axis ticks
    ax.set_title(title)

    plt.show()


#%%Average Kinetic Plot
def average_decay(data, x=-1000, y=-1, units='ns', label='label me', 
                  plot=True,marker='o', wlmin=2, wlmax=-4, title=None,tEnd = -1, 
                  linestyle=None,tcorrect=True, normalise=False, ax=None, maxTime=-1, 
                  xaxis='symlog', yaxis='linear', tcorrectadjust=0,constant=0,legendTitle='Legend',yMin=.00000001):
    selected_wls = data[:, wlmin:wlmax] # Selects the wavelengths of interest
    print(selected_wls)
   
    time = data[1:maxTime, 0].round(1) # Selects the timepoints of interest
    time_1 = np.where(time == 1)[0][0] # Finds the index of the timepoint 1 ns (close enough to max signal)
    pos_signal = np.where(y * selected_wls[time_1,:] > 0)[0] # Finds the indices of the wavelengths with a positive signal
    signal = x * np.mean(selected_wls[1:maxTime, pos_signal], axis=1)  # Averages the selected wavelengths and multiplies by x
    
    
    
    max_signal = signal.argmax() + tcorrectadjust # Finds the index of the maximum signal
    if tcorrect:
        time = time[max_signal:] - time[max_signal] #Corrects time to start at 0
        signal = signal[max_signal:] #Corrects signal to start at max signal
    
    if normalise:
        signal = signal / signal[0] # Normalises signal to start at 1
    new_signal = np.zeros(len(signal))
    for i,s in enumerate(signal):
        new_signal[i] = s+abs(signal.min())+constant
    decay = np.array((time, new_signal)) # Combines time and signal into one array
    
    if plot ==True:
        fig, ax = plt.subplots() if ax is None else (None, ax)
        ax.plot(decay[:,0], decay[:,1]+constant, marker,alpha=0.6,label=label)
        ax.set_yscale(yaxis)
        ax.set_xscale(xaxis)
        ax.set_title(title)
        ax.set_xlabel(f'Time/ {units}')
        ax.set_ylabel(f'${{\Delta}}$T/T{" Norm." if normalise else ""}')
        #ax.set_xlim(time[0], time[tEnd])
        if normalise:
            ax.set_ylim(yMin,1)
        ax.legend(title=legendTitle)
    
    return decay
#%% Align drift in time zero onset

def align_rise(data,plot=True,save=False,name='YOur File Name', endTime = -1,y_correct=False,fromMax=False,constant=0):
    timepoints = data[:endTime, 0]      # Define timepoints
    signals = data[:endTime, 1:]        # Define signals
    unaligned_mean = signals.mean(axis=1)  # Define the mean signal of unaligned
    target = 0.5*unaligned_mean.max()      # Define a target value on the rise

    aligned_array = np.zeros((timepoints.shape[0],data.shape[1]))
    aligned_array[:,0] = timepoints
    
    
    for i in range(0,signals.shape[1]):
        signal = signals[:,i] #Selects sweep
        
        closest_index = np.argmin(np.abs(signal[:signal.argmax()] - target))            # Finds closest index to target value
        
        closest_timepoint = timepoints[closest_index]                                   # Finds the time of the corresponding index

        new_timepoints = timepoints - closest_timepoint                                 # shift timepoints
        
        function = interpolate.interp1d(new_timepoints,signal,fill_value='extrapolate') # makes function for new timepoints and signal
        
        aligned_array[:,(i+1)] = function(timepoints)                                   # interpolates original timepoints into function 

        
    aligned_mean = np.mean(aligned_array[:,1:],axis=1)
    aligned_mean = aligned_mean+abs(aligned_mean.min())+constant
    

    if plot:
        colormap = plt.cm.get_cmap('brg', signals.shape[1])
        fig, ax = plt.subplots(2, 2, sharex=True)
    
        for i in range(signals.shape[1]):
            ax[0, 0].plot(timepoints, signals[:, i], '.', c=colormap(i), alpha=0.3)
            ax[0, 1].plot(timepoints, aligned_array[:, 1+i], '.', c=colormap(i), alpha=0.3, label=i+1)
            ax[1, 0].plot(timepoints, unaligned_mean, 'k.')
            ax[1, 1].plot(timepoints, aligned_mean, 'k.')
    
        fig.suptitle(name, fontsize=14)
        ax[0, 0].set_title('Before')
        ax[0, 0].set_ylabel(f"""Individual Sweeps
        $\Delta$T/T x 10$^{-3}$""")
        ax[0, 1].set_title('After')
        
        if signals.shape[1] > 10:
            fig.subplots_adjust(right=0.8)  # make room for colorbar
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # define colorbar axis
            norm = mpl.colors.Normalize(vmin=1, vmax=signals.shape[1]+1)  # define color normalization
            cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=colormap, norm=norm, orientation='vertical',label='Sweep Number')  # create colorbar
            for i in range(signals.shape[1]):
                ax[0, 0].plot(timepoints, signals[:, i], '.', c=colormap(i, alpha=0.3))
                ax[0, 1].plot(timepoints, aligned_array[:, 1+i], '.', c=colormap(i, alpha=0.3))
        else:
            ax[0, 1].legend(loc='upper right', ncol=1 if signals.shape[1] < 11 else 3, fontsize='xx-small')
    
        ax[1, 0].set_ylabel("""Averaged
        $\Delta$T/T x 10$^{-3}$""")
        ax[1, 1].set_xlabel('Time/ ns')
        ax[1, 0].set_xlabel('Time/ ns')
        for a in ax.flatten():
            a.set_xscale('symlog', linthresh=10)
            a.tick_params(axis='x', labelsize=8)

        if save:
            if name == None:
                print('Call name in function to name file')
            else:
                plt.savefig('Alignment' + name[:-5] + '.svg')
    plt.show()
    if fromMax:
        
        signal_max = aligned_mean.argmax()
        timepoints = timepoints[signal_max:]-timepoints[signal_max]
        aligned_mean = aligned_mean[signal_max:]
    plt.plot(timepoints,aligned_mean,'b.')
    plt.xscale('symlog',linthresh=10)
    plt.yscale('log')
    plt.xlabel('Time/ ns')
    plt.ylabel('$\Delta$T/T x 10$^{-3}$')
    plt.title(f'Aligned Average - {name}')
    
    aligned_average = np.array((timepoints,aligned_mean))
    
    return aligned_average


#%% All Sweeps at once

def sweep_decays(arr3d, endTime=-1, wlmin=100, wlmax=-50, x=1000, title='All Sweeps',plot=True, 
               xaxis='symlog',marker='-',normalise=False,remove_zeros = False, good=False,constant=0,
               noise_threshold=None,exclude_sweeps=[]):
    time = arr3d[0, 1:endTime, 0]                   # Define timepoints
    data = time[:, np.newaxis]                      # Make timepoints 2D
    
    decays = np.zeros((len(time), 0))               # Create empty array for decays
    for i, sweep in enumerate(arr3d):
        if i in exclude_sweeps:                         # Excludes sweeps in exclude_sweeps list
                continue                                # Skips to next sweep
        
        arr2d2 = arr3d[i, 1:endTime, wlmin:wlmax]   # Selects sweep
        signal = x*np.mean(arr2d2[:, :], axis=1)    # Averages wavelengths and multiplies by x
        if remove_zeros:                            # Removes zeros from signal
            zero_vals = np.where(signal == 0)[0]
            for z in zero_vals:
                signal[z] = (signal[z-1]+signal[z+1])/2 # Replaces zeros with average of surrounding values
        if normalise:
            signal = signal/signal.max()            # Normalises signal to start at 1
            
        if noise_threshold is not None:             # Removes noise from signal
            for j in range(1, len(signal) - 1):     
                prev_val = signal[j - 1]        # Defines previous values
                next_val = signal[j + 1]        # Defines next values
                if abs(signal[j] - prev_val) > noise_threshold * abs(prev_val - next_val) or \
                abs(signal[j] - next_val) > noise_threshold * abs(prev_val - next_val): # If the difference between the current value and the previous/next value is greater than the noise threshold times the difference between the previous and next value
                    signal[j] = (prev_val + next_val) / 2
                
        decays = np.append(decays, signal[:, np.newaxis], axis=1) # Adds signal to decays array
        
    sweepDecays =  np.hstack((data, decays))      # Combines time and signal into one array
    if plot:                                     # Plots decays
        fig, ax = plt.subplots()
        colormap = plt.cm.get_cmap('brg', sweepDecays.shape[1])
        for i in range(1,sweepDecays.shape[1]):
            ax.plot(sweepDecays[:,0],sweepDecays[:,i]+constant,marker, c=colormap(i-1), label=f"{i}")
            ax.set_xscale(xaxis)
            ax.set_yscale('linear')
            ax.set_xlabel('Time/ ns')
            ax.set_ylabel('$\Delta$T/T x 10$^{-3}$')
            ax.set_title(title)
            if sweepDecays.shape[1] < 9:
                ax.legend(loc='upper right',title = 'Sweep no.')
            elif sweepDecays.shape[1] >= 9 and sweepDecays.shape[1] < 25:
                ax.legend(loc='upper right', title = 'Sweep no.', ncol=2, fontsize = 'small')
            else:
                ax.legend(bbox_to_anchor=(1, -.15), title = 'Sweep no.', ncol=7, fontsize = 'small')
    plt.show()
    if good:                                    # Plots decays with good sweeps
        mean_signal = np.mean(sweepDecays[:, 1:], axis=1)[:, np.newaxis]
        print(mean_signal.shape)
        mean_decay = np.hstack((data, mean_signal))
        return mean_decay
    else:
        return sweepDecays


#%% Single Sweeps
def sing_sweep(file,sweepNo=0,x=1000,wlmin=1,wlmax=-1,maxTime=None,y=-1,normalise=False,timecorrect = False):
    arr3d = file
    arr3d_2 = arr3d[:,0:,wlmin:wlmax]
    time = arr3d[sweepNo,1:maxTime,0].round(1)
    timeOne = np.where(time==1)[0][0]
    findPosSig = np.where(y*arr3d_2[sweepNo,timeOne,:]>0)[0]
    posSig = np.mean(y*arr3d_2[sweepNo,1:maxTime,findPosSig],axis=0)
    if timecorrect == True:
        maxSig = posSig.argmax()
        time = time[maxSig:]-time[maxSig]
        sig = posSig[maxSig:]
    else:
        sig=posSig
    norm = sig/sig.max()
    fig, ax = plt.subplots()
    if normalise == False:
        ax.plot(time,x*sig)
    else:
        ax.plot(time,norm)
    A = int(np.log10(1/x))
    ax.set_xscale('symlog',linthresh=1)
    ax.set_xlabel('Time/ ns')
    ax.set_ylabel(f'${{\Delta}}$T/T x 10$^{{{A}}}$')
    ax.set_xlim(time.min(),time.max())
    return arr3d
    
#file = 'TC3_1_500us_30uW.hdf5'
#sing_sweep(file, sweepNo=3,wlmin=90,wlmax = -100, maxTime = -20,normalise=False, timecorrect = False)
   
#%% List Comprehension
def order_list(path=None,fileName = 'choose part of file name',labLeft=None,labRight=None,integer=None):
    filepath = path
    filelist = [file for file in os.listdir(path) if file.startswith(fileName)]
    if integer == True:
        filelist.sort(key = lambda x: int(x[labLeft:labRight]))
    else:
        filelist.sort(key = lambda x: x[labLeft:labRight])
    return filelist

#%% Plot Spectrum
def plot_spec(arr2d,time=[1], x=-1000, wlmin=1, wlmax=-1, labelUnit = ' ns', tUnit = 'ns',title='yourtitle',ax=None, eV = False):

    wavelengths = arr2d[0,wlmin:wlmax]
    timepoints = arr2d[1:, 0].round(1).tolist()
                 
    if ax==None:
        fig, ax = plt.subplots()
        
    for t in time:    

        timeIndex = timepoints.index(t)
        signal = x*arr2d[timeIndex,wlmin:wlmax]
        if eV == True:
            xdata = h*c/(e*wavelengths*1e-9)
            signal = signal*(h*c/((xdata)**2))
            ax.plot(xdata,signal, label =str(t)+labelUnit)
            ax.set_xlabel('eV')
        else:
            xdata = wavelengths
            ax.plot(xdata, signal, label = str(t) + labelUnit)
            ax.set_xlabel('Wavelengths (nm)')
        A = int(np.log10(abs(1/x)))
        ax.set_ylabel(f'${{\Delta}}$T/T x 10$^{{{A}}}$')
        ax.set_xlim(xdata.min(),xdata.max())
        ax.axhline(y=0,color='grey')
        ax.minorticks_on()
        ax.legend()
        ax.set_title(title)

    plt.show()
        
    return arr2d


#%%
def crop_av(file,delRows=False,rows=-1,delCols=False,cols=-1):
    arr2d = mean_sweeps(file)
    if delRows==True and delCols==False:
        arr2d_2 = np.delete(arr2d,[rows],axis=0)
    elif delRows==False and delCols==True:
        arr2d_2 = np.delete(arr2d,[cols],axis=1)
    else:
        arr2d_1 = np.delete(arr2d,[rows],axis=0)
        arr2d_2 = np.delete(arr2d_1,[cols],axis=1)
    return arr2d_2



#%% Carrier Density

def calc_fluences(P=[1,5,10],A = 1, d = 500, WL = 400, di = 1500, f=500,areaType='gauss',a=2500,b=1950):
    hv = (6.626E-34*2.998E8)/(WL*1e-9)                    #Calculate photon energy
    absorption = 1-10**(-A)                               #Calculate % of Photons absorbed
    if areaType == 'elliptical':                          #Calculate area in cm3 based on shape
        area = np.pi*((a*1e-4)/2)*((b*1e-4)/2)            #Ellipse is pi*rad1*rad2
    else:
        area = np.pi*(di*1e-4/2)**2                       #Circle pi*r^2
   
    nlist = []
    for p in P:
        n = (absorption*p*1e-6)/(area*f*hv*d*1e-7)        # in cm-3
        
        nlist.append(n)
    
    pflist = []
    for p in P:
        fl = (p)/(area) #uW/cm-2
        
        pflist.append(fl)
        
    eflist = []
    for p in P:
        el = (p)/(area*f) #uW/cm-2*1000
        
        eflist.append(el)
    print(f'Carrier Densities in cm\u207B\u00B3 = {nlist}')
    print(f'Power fluence in \u03BCWcm\u207B\u00B2 = {pflist}')
    print(f'Energy fluence in nWcm\u207B\u00B2 = {eflist}')
        
    return nlist, pflist, eflist
 #%% Save Data
 
def save_TA_data(filename, variable):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)





    