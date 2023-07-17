###################################################################################### FUNCTIONS
###################################################################################### 
###################################################################################### 
###################################################################################### 
###################################################################################### 


def event_extraction(config,tsvfile,dyad):
    # What kind of events do you want? These should be the EXACT text of what is in the tsv column
    eventtype_bra = config["event_descriptions"]["brake"]
    eventtype_ll = config["event_descriptions"]["left_lane"]
    eventtype_rl = config["event_descriptions"]["right_lane"]
    eventtype_acc = config["event_descriptions"]["acceleration"]
    
    # Initialization  # create a blank list of event times in [s]
    eventtimes_bra = []; eventtimes_ll = []; eventtimes_rl = []; eventtimes_acc = [] 
    # NEW
    tsv_dataframe = pd.read_csv(tsvfile.name, sep="\t", header=None)
    tsv_dataframe.rename({1:'eventtime', 2:"description"},axis=1,inplace=True)

    eventtimes_bra = tsv_dataframe[tsv_dataframe.apply(lambda r: eventtype_bra in r['description'], axis=1)]["eventtime"].to_list()
    eventtimes_ll = tsv_dataframe[tsv_dataframe.apply(lambda r: eventtype_ll in r['description'], axis=1)]["eventtime"].to_list()
    eventtimes_rl = tsv_dataframe[tsv_dataframe.apply(lambda r: eventtype_rl in r['description'], axis=1)]["eventtime"].to_list()
    eventtimes_acc = tsv_dataframe[tsv_dataframe.apply(lambda r: eventtype_acc in r['description'], axis=1)]["eventtime"].to_list()
        
    count_bra=len(eventtimes_bra)
    count_ll=len(eventtimes_ll)
    count_rl=len(eventtimes_rl)
    count_acc=len(eventtimes_acc)
    orig_num_events=count_bra+count_ll+count_rl+count_acc
    f.write("-The Total Number of Braking Events = %s \n" %count_bra)
    f.write("-The Total Number of Left-Lane Events = %s \n" %count_ll)
    f.write("-The Total Number of Right-Lane Events = %s \n" %count_rl)
    f.write("-The Total Number of Acceleration Events = %s \n" %count_acc)
    f.write("-Total Number of Events =  %s \n" %orig_num_events)
    
    # Generating no-event events 
    window_size = 10                                                     # in [s] (choose it as big as possible and then any smaller segment does not include any events automatically)
    count_nov=  math.floor((count_bra+count_ll+count_rl+count_acc)/4)  # Avg. number of events is how many no-event events needed
    artificial_eventID_col= np.ones(shape=(10800,1))                   # Three-hour array of 1-s samples
    
    #for i in range(0,len(artificial_eventID_col)):
    #   if i in np.rint(eventtimes_bra):
    #        artificial_eventID_col[i,0]=5
    #   elif i in np.rint(eventtimes_ll):
    #        artificial_eventID_col[i,0]=2
    #   elif i in np.rint(eventtimes_rl):
    #        artificial_eventID_col[i,0]=4
    #   elif i in np.rint(eventtimes_acc):
    #        artificial_eventID_col[i,0]=3
    
    #artificial_eventID_col_pd=pd.DataFrame(artificial_eventID_col,columns=["ID"])
    #pre_drive_index = artificial_eventID_col_pd[artificial_eventID_col_pd["ID"] != 1].index[0]
    #post_drive_index = artificial_eventID_col_pd[artificial_eventID_col_pd["ID"] != 1].index[-1]

    ## Actual generation
    #i=1
    #temp_time_nov=[]
    #while i<=count_nov:
    #    random_beg=random.randint(pre_drive_index+window_size,post_drive_index)                  # pick a random time(or index as they are equal assuming 1 sample/s) during the drive
    #    if np.sum(artificial_eventID_col_pd[random_beg-window_size:random_beg].ID)==window_size: # if all samples are no-event, then take this segment
    #        temp_time_nov.append(random_beg)                                                     # list of indicies indicating the start of no-event events
    #        i+=1        

    # to get a new recording, uncomment above and delete next row. This is only done for reproducibility.
    # Do the below temp_time_nov have the event ids where no event is occuring? Are the taken manually? 
    # and do the above lines of code tries to get them automatically ?
    if dyad == 5:
        temp_time_nov=[3301, 5249, 5274, 5017, 4072, 3189, 2864, 4605, 2597, 3328, 2664, 5410, 3837, 3595, 3750, 2336, 3361, 4279, 3827, 5429, 4136, 5331, 5025, 3263, 4483, 3134, 4774, 3813, 4889, 2685, 5227, 2620, 4174, 5249, 4030, 3289, 3061, 5197, 3356, 4682, 4474, 5154, 3887, 3649, 3604, 4807, 3144, 3601, 4394, 5335, 5243]
    elif dyad == 10:
        temp_time_nov=[2450, 2049, 2692, 2427, 2284, 4458, 3785, 2077, 2475, 2943, 4158, 3198, 4007, 3671, 2413, 1758, 1983, 3943, 3232, 3730, 2201, 2642, 1703, 2694, 2901, 4757, 2910, 3625, 2981, 4727, 3249, 4651, 4509, 3780] 
    elif dyad == 1:
        temp_time_nov=[2649, 5445, 2903, 5023, 3693, 3108, 3639, 4182, 4434, 4308, 3718, 4623, 4818, 2500, 4369, 4826, 5348, 4450, 4508, 2587, 2908, 4572, 4158, 2552, 4267, 3913, 5036, 5106, 2924, 5527, 3576, 4445, 3408, 2404, 4607, 2889]
    #
    eventtimes_nov= np.rint(temp_time_nov)
    tot_num_events=orig_num_events+count_nov

    f.write("-Number of No-Event Events (Avg. Num. of Other Events) = %s \n" %count_nov)
    f.write("-Number of Total Events (Including No-Event Events) = %s \n" %tot_num_events)

    # Driving events
    event_dict = {   
    5:{"name":"Braking","times":eventtimes_bra},
    2:{"name":"Left_Turn","times":eventtimes_ll},
    4:{"name":"Right_Turn","times":eventtimes_rl},
    3:{"name":"Acceleration","times":eventtimes_acc},
    1:{"name":"NoEv","times":eventtimes_nov}
                }
    
    # Counts Dict
    counts_dict = {
                   "braking": count_bra,
                   "left_lane":count_ll,
                   "right_lane":count_rl,
                   "acceleration": count_acc,
                   "no_events":count_nov
                }
    
    # return event_dict,count_bra,count_ll,count_rl,count_acc,count_nov,orig_num_events,tot_num_events
    return event_dict,counts_dict,tot_num_events

def modality_synchronization(config,dyads_list,dyad,event_dict):

    # This is the time difference between event file and data files
    # The convention is: time in data file=time in event file + synch       # +VE synch: event file is lagging, -VE synch: event file is leading

    # For EEG and ECG: No way to find the synchronization variable manually, so:
    # 1) Found the total length of the data file
    # 2) Found the total length of the event file
    # 3) Aligned both files from head and tail and kept the one that yielded better results (after trying out all possible cases, we will take zero difference in synchronization as these generally provide the highest and most consistent results for all subjects )

    EEG_freq=config["freq"]["eeg"]
    Eye_freq=config["freq"]["eye"]
    Seat_freq=config["freq"]["seat"]
    ECG_freq=config["freq"]["ecg"]
    EEG_synch_diff={  
        dyads_list[0]: config["synch_diff"]["eeg"]["d1"],          
        dyads_list[1]: config["synch_diff"]["eeg"]["d10"],      
        dyads_list[2]: config["synch_diff"]["eeg"]["d5"]      
        }

    ECG_synch_diff={  
        dyads_list[0]: config["synch_diff"]["ecg"]["d1"],         
        dyads_list[1]: config["synch_diff"]["ecg"]["d10"],       
        dyads_list[2]: config["synch_diff"]["ecg"]["d5"]      
        }

    # For Eye, the synch variable is found manually by comparing the data file with the event file directly
    Eye_synch_diff={  
        dyads_list[0]: config["synch_diff"]["eye"]["d1"],   # This number is not accurate     
        dyads_list[1]: config["synch_diff"]["eye"]["d10"],  
        dyads_list[2]: config["synch_diff"]["eye"]["d5"]    # This number is not accurate
        }

    # For Seat, synchronization is done through the video as common element between data file and event file
    # The two equations are: 
    # 1) Time in data file = time in video + synch1
    # 2) Time in event file + synch2 = time in video file. Synch2 here is the same as Eye_synch_diff
    # By solving 1) and 2), we get: 3) Time in data file = time in event file + synch1 + synch2 

    Seat_video_synch_diff={     # These are (synch1), between data and video files
        dyads_list[0]: config["synch_diff"]["seat_video"]["d1"],       
        dyads_list[1]: config["synch_diff"]["seat_video"]["d10"],        
        dyads_list[2]: config["synch_diff"]["seat_video"]["d5"]
        }

    Seat_synch_diff={           # These are (synch1+synch2), between data and event files
    dyads_list[0]: config["synch_diff"]["seat"]["d1"],      
    dyads_list[1]: config["synch_diff"]["seat"]["d10"],
    dyads_list[2]: config["synch_diff"]["seat"]["d5"]
    }

    # The baseline period is 60-120s at the beginning of the drive (found manually)
    # These are the values as read from the video
    baseline_start_time={  
        dyads_list[0]: config["baseline"]["start_time"]["d1"],       
        dyads_list[1]: config["baseline"]["start_time"]["d10"],
        dyads_list[2]: config["baseline"]["start_time"]["d5"]
        }

    baseline_end_time={  
        dyads_list[0]: config["baseline"]["end_time"]["d1"],      
        dyads_list[1]: config["baseline"]["end_time"]["d10"],        
        dyads_list[2]: config["baseline"]["end_time"]["d5"]
        }

    ### TO DO: reate common function that takes synch_diff and freq as inputs 
    #   and excecute the code in for loop
    # NEW
    def create_event_index(freq, synch_diff):
        event_indexes_dict={}
        for event_type in event_dict:
            event_index=np.add(np.array(event_dict[event_type]["times"]),synch_diff[dyad])
            event_index=np.multiply(event_index,freq)
            event_index=np.rint(event_index)
            event_index=np.expand_dims(event_index, axis=1)
            event_indexes_dict[event_type]=event_index; del event_index
        return event_indexes_dict

    event_indexes_dict_EEG = create_event_index(EEG_freq, EEG_synch_diff)
    event_indexes_dict_ECG = create_event_index(ECG_freq, ECG_synch_diff)
    event_indexes_dict_Eye = create_event_index(Eye_freq, Eye_synch_diff)
    event_indexes_dict_Seat = create_event_index(Seat_freq, Seat_synch_diff)
    
    event_indexes_dict={
        "EEG" : event_indexes_dict_EEG,
        "ECG" : event_indexes_dict_ECG,
        "Eye" : event_indexes_dict_Eye,
        "Seat" : event_indexes_dict_Seat
    }

    return(event_indexes_dict,Seat_video_synch_diff,baseline_start_time,baseline_end_time)

def feature_extraction(config,data_files_dict,dyad,event_dict,event_indexes_dict,tot_num_events,Seat_video_synch_diff,baseline_start_time,baseline_end_time):

    EEG_freq=config["freq"]["eeg"]
    Eye_freq=config["freq"]["eye"]
    Seat_freq=config["freq"]["seat"]
    ECG_freq=config["freq"]["ecg"]

    EEG_data=data_files_dict["eeg"]
    Eye_data=data_files_dict["eye"]
    Seat_data=data_files_dict["seat"]
    ECG_data=data_files_dict["ecg"]

    EEG_window_size=config["window_size"]["eeg"]     # in [s]
    Eye_window_size=config["window_size"]["eye"]     # in [s]
    Seat_window_size=config["window_size"]["seat"]     # in [s]
    ECG_window_size=config["window_size"]["ecg"]      # in [s] 
    
    event_indexes_dict_EEG = event_indexes_dict["EEG"]
    event_indexes_dict_ECG = event_indexes_dict["ECG"]
    event_indexes_dict_Eye = event_indexes_dict["Eye"]
    event_indexes_dict_Seat = event_indexes_dict["Seat"]
        # EEG
        # 1)Prep for MNE (getting the events file in needed format)
    print("EEG preprocessing is in progress (may take up to 1 min)")
    ready_events=np.zeros(shape=(1,3))
    for event_type in event_indexes_dict_EEG:
        eventID= event_dict[event_type]
        for event in event_indexes_dict_EEG[event_type]:
            my_index=event
            temp_events=[int(my_index),int(0),event_type]
            temp_events=np.reshape(temp_events,(1,3))
            ready_events=np.append(ready_events,temp_events,axis=0)
    ready_events=np.delete(ready_events, 0, axis=0).astype('int')    

         # 2)Define MNE elements


    chosen_channels = list(config["mne"]["channel_types"].keys())   # we'll remove the occipital electrodes from being used in the classification
    info = mne.create_info(ch_names=chosen_channels, sfreq=EEG_freq)  # this is how MNE understands data
    raw = mne.io.RawArray(EEG_data[0:-3], info)
    raw.set_channel_types(config["mne"]["channel_types"])
    raw.set_eeg_reference('average',ch_type='eeg')                    # CAR for averaging
    raw.filter(config["mne"]["alpha_beta_filter"][0], 
               config["mne"]["alpha_beta_filter"][1], 
               fir_design='firwin', 
               skip_by_annotation='edge') # filtration-keep alpha and beta
    montage = mne.channels.make_standard_montage('standard_1020')    # Montage is only done to genearte head plots
    raw.set_montage(montage)

         # 3) ICA
    noise_covarience = mne.compute_raw_covariance(raw,tmin=2000, tmax=2120) 
    ica = ICA(n_components=config["ica"]["n_components"], max_iter='auto', random_state=97, noise_cov=noise_covarience)
    ica.fit(raw)     
    ica.exclude = config["ica"]["exclude"]   # 1 is eye movement
    ica.apply(raw,exclude=config["ica"]["exclude"])
    #ica.plot_components()
                            

    #mod_count_nov=len(my_labels[my_labels==1]);mod_count_bra=len(my_labels[my_labels==5]);mod_count_ll=len(my_labels[my_labels==2]);mod_count_rl=len(my_labels[my_labels==4]);mod_count_acc=len(my_labels[my_labels==3])
    #print("EEG preprocessing is completed \n","-----------")
    event_id = dict(No_event=1, Left_turn=2, Accelaration=3, Right_turn=4, Braking=5) 
    tmin, tmax = -EEG_window_size+.25,-.25
    events=ready_events
    rejection=None#dict(eeg=60)            # in mV
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=chosen_channels,  preload=False,proj=True, baseline=None,reject=rejection,event_repeated="merge",reject_by_annotation=False) 
    epochs_data = epochs.get_data()
    retained_epochs= epochs.selection
    my_labels = epochs.events[:, -1]
    mod_count_nov=len(my_labels[my_labels==1])
    mod_count_bra=len(my_labels[my_labels==5])
    mod_count_ll=len(my_labels[my_labels==2])
    mod_count_rl=len(my_labels[my_labels==4])
    mod_count_acc=len(my_labels[my_labels==3])
    print("EEG preprocessing is completed \n","-----------")



        # 5) Feature extraction
    EEG_features=[]
    for i in range (0,len(my_labels)):   # loop over epochs
        my_data=epochs[i].get_data()
        my_psd_current = psd_array_welch(my_data,fmin=8, fmax=38, sfreq=EEG_freq,n_per_seg=None,n_fft=256,average="mean")
        my_psd_current = np.array(my_psd_current[0]) 
        my_psd_current=my_psd_current[0]
        transposed_arr = np.transpose(my_psd_current)
        frontal_psd= transposed_arr[:,0:5]
        central_psd= transposed_arr[:,5:11]
        parietal_psd= transposed_arr[:,11:17]
        avg_frontal_psd= np.mean(frontal_psd, axis=1)
        avg_central_psd= np.mean(central_psd, axis=1)
        avg_parietal_psd= np.mean(parietal_psd, axis=1)
        
        avg_rhythm_frontal_psd=[]
        for i in range (1,len(avg_frontal_psd),5):
            temp_var= np.mean(avg_frontal_psd[i:i+5])
            avg_rhythm_frontal_psd.append(temp_var)

        avg_rhythm_central_psd=[]
        for i in range (1,len(avg_central_psd),5):
            temp_var= np.mean(avg_central_psd[i:i+5])
            avg_rhythm_central_psd.append(temp_var)

        avg_rhythm_parietal_psd=[]
        for i in range (1,len(avg_parietal_psd),5):
            temp_var= np.mean(avg_parietal_psd[i:i+5])
            avg_rhythm_parietal_psd.append(temp_var)

        feature_vector=avg_rhythm_frontal_psd+avg_rhythm_central_psd+avg_rhythm_parietal_psd
        feature_vector_np=np.array(feature_vector)
        EEG_features.append(feature_vector_np)

    
    print("EEG features are extracted. \n","-----------")
    f.write("-Considered EEG features are: Avg. PSD for a) Alpha, b) Beta1, c) Beta2, d) Beta3, e) Gamma1, and f) Gamma2 for 1) frontal, 2) central, and 3) parietal \n")

    # Eye
    num_front=[]; num_left=[]; num_right=[]
    Eye_window_size_samples=Eye_window_size*Eye_freq
    Eye_features=[]
    for event_type in event_indexes_dict_Eye:
        for event in event_indexes_dict_Eye[event_type]:
            my_index=int(event)
            modified_index=Eye_data.index[Eye_data["image_ID"] == "frame%s.jpg" %my_index].tolist() # because of dropped frames
            modified_index=int(modified_index[0])
            my_window_labels=Eye_data["Class"][modified_index-Eye_window_size_samples:modified_index]  # find the labels corresponding to this timepoint
            num_front=my_window_labels[my_window_labels==0].count()  # add the frames (seconds) where the passenger was looking straight
            num_left=my_window_labels[my_window_labels==1].count()   # add the frames (seconds) where the passenger was looking left
            num_right=my_window_labels[my_window_labels==2].count()  # add the frames (seconds) where the passenger was looking right
            Eye_features.append([num_front,num_left,num_right])        
    print("Eye features are extracted. \n","-----------")
    f.write("-Considered Eye features are: 1) Percentage of looking straights, 2) Percentage of looking right, and 3) Percentage of looking left \n" )

    # Seat
    arr_sensors=np.array(Seat_data)     # the raw data
    beg_baseline_period=(baseline_start_time[dyad]+ Seat_video_synch_diff[dyad])*Seat_freq        
    end_baseline_period=(baseline_end_time[dyad]+ Seat_video_synch_diff[dyad])*Seat_freq
    for i in range(0,12):               
        baseline_sensor=np.round(np.average(arr_sensors[beg_baseline_period:end_baseline_period,i]))
        baseline_sensor=baseline_sensor.astype('int64')     # remove the baseline readings
        arr_sensors[:,i] -=baseline_sensor
        low_percentile=np.percentile(arr_sensors[:,i],5)   # remove the outliers
        high_percentile=np.percentile(arr_sensors[:,i],95)
        for c in range(0,len(arr_sensors[:,i])):
            value=  arr_sensors[c,i]
            if value <  low_percentile:
               arr_sensors[value,i]=  low_percentile
            elif value >  high_percentile:
               arr_sensors[value,i]=  high_percentile
   
    my_window_data=np.ndarray(shape=(Seat_window_size*Seat_freq,12))
    Seat_window_size_samples=Seat_window_size*Seat_freq
    Seat_features=[]
    for event_type in event_indexes_dict_Seat:
        for event in event_indexes_dict_Seat[event_type]:
            my_index=int(event)
            my_window_labels=arr_sensors[my_index-Seat_window_size_samples:my_index]  
            
            ##### Features
            avg_s1=np.mean(my_window_labels[:,0]); std_s1=np.std(my_window_labels[:,0]); med_s1=np.median(my_window_labels[:,0])
            avg_s2=np.mean(my_window_labels[:,1]); std_s2=np.std(my_window_labels[:,1]); med_s2=np.median(my_window_labels[:,1])
            avg_s3=np.mean(my_window_labels[:,2]); std_s3=np.std(my_window_labels[:,2]); med_s3=np.median(my_window_labels[:,2])
            avg_s4=np.mean(my_window_labels[:,3]); std_s4=np.std(my_window_labels[:,3]); med_s4=np.median(my_window_labels[:,3])
            avg_s5=np.mean(my_window_labels[:,4]); std_s5=np.std(my_window_labels[:,4]); med_s5=np.median(my_window_labels[:,4])
            avg_s6=np.mean(my_window_labels[:,5]); std_s6=np.std(my_window_labels[:,5]); med_s6=np.median(my_window_labels[:,5])
            avg_s7=np.mean(my_window_labels[:,6]); std_s7=np.std(my_window_labels[:,6]); med_s7=np.median(my_window_labels[:,6])
            avg_s8=np.mean(my_window_labels[:,7]); std_s8=np.std(my_window_labels[:,7]); med_s8=np.median(my_window_labels[:,7])
            avg_s9=np.mean(my_window_labels[:,8]); std_s9=np.std(my_window_labels[:,8]); med_s9=np.median(my_window_labels[:,8])
            avg_s10=np.mean(my_window_labels[:,9]);std_s10=np.std(my_window_labels[:,9]); med_s10=np.median(my_window_labels[:,9])
            avg_s11=np.mean(my_window_labels[:,10]); std_s11=np.std(my_window_labels[:,10]); med_s11=np.median(my_window_labels[:,10])
            avg_s12=np.mean(my_window_labels[:,11]); std_s12=np.std(my_window_labels[:,11]); med_s12=np.median(my_window_labels[:,11])

            Seat_features.append([avg_s1,avg_s2,avg_s3,avg_s4,avg_s5,avg_s6,avg_s7,avg_s8,avg_s9,avg_s10,avg_s11,avg_s12,std_s1,std_s2,std_s3,std_s4,std_s5,std_s6,std_s7,std_s8,std_s9,std_s10,std_s11,std_s12,med_s1,med_s2,med_s3,med_s4,med_s5,med_s6,med_s7,med_s8,med_s9,med_s10,med_s11,med_s12])
    print("Seat features are extracted. \n","-----------")
    f.write("-Considered Seat features are: 1) Avg., 2)Std, and 3) median values, for the 12 sensors \n")

    #ECG
    print("ECG features are being extracted (may take up to 1 min)","\n")
    filtered_ECG_data = hp.filter_signal(ECG_data, cutoff = [0.20, 5], sample_rate = ECG_freq, order = 3, filtertype='bandpass')
    drdata, measures=hp.process(filtered_ECG_data,sample_rate=250, report_time=True)
    ECG_window_size_samples=ECG_window_size*ECG_freq
    ECG_features=[]
    drop_ecg=[]
    c=0
    for event_type in event_indexes_dict_ECG:
        for event in event_indexes_dict_ECG[event_type]:
            my_index=int(event)
            my_window_labels=filtered_ECG_data[my_index-ECG_window_size_samples:my_index]  # find the labels corresponding to this timepoint
            filtered_window_ECG_data, measures=hp.process(my_window_labels,sample_rate=ECG_freq, report_time=False)
            my_bpm=measures['bpm']
            my_rmssd=measures['rmssd']  
            my_ibi=measures['ibi']
            my_sdnn=measures['sdnn']
            my_sdsd=measures['sdsd']
            my_pnn20=measures['pnn20']
            my_pnn50=measures['pnn50']
            if math.isnan(my_bpm)==True or math.isnan(my_rmssd)==True or math.isnan(my_ibi)==True or math.isnan(my_sdnn)==True or math.isnan(my_sdsd)==True or math.isnan(my_pnn20)==True or math.isnan(my_pnn50)==True:
                drop_ecg.append(c)
            ECG_features.append([my_bpm,my_rmssd,my_ibi,my_sdnn,my_sdsd,my_pnn20,my_pnn50])
            c+=1
    print("ECG features are extracted. \n")
    f.write("-Considered ECG features are: 1)bpm, 2)rmssd, 3)ibi, 4)sdnn, 5)sdsd, 6)pnn20, and 7)pnn \n")

    # Deletion of dropped epochs by ECG 
    to_delete=[]
    for i in range(0,tot_num_events):
        if i in drop_ecg:
            to_delete.append(i)

    for index in sorted(to_delete, reverse=True):
        del ECG_features[index];
        del Eye_features[index];
        del Seat_features[index];
        del EEG_features[index];
        my_labels=np.delete(my_labels,index)


    for i in range(0,len(my_labels)):       # Merging both lane changes as one event
        if my_labels[i]==4:
            my_labels[i]=2

    f.write("-Total number of events after dropping bad epochs is: %s \n" %len(my_labels))
    f.write("-Classifier used: SVM-rbf (5CV) \n") 
    f.write("#####################################") 
    
    feature_dict = { 
        "EEG"  : EEG_features,
        "ECG"  : ECG_features,
        "Eye"  : Eye_features,
        "Seat" : Seat_features
        }
    return feature_dict, my_labels

def unimodal_classification(config,feature_dict,my_labels):
        # feature_dict={ "EEG"  : EEG_features,
        #            "ECG"  : ECG_features,
        #            "Eye"  : Eye_features,
        #            "Seat" : Seat_features
        #             }
    ##### Single split
    ## Merge both for loops.
    ## modality_name variable is not needed.
    ## y_test should be the first variable in confusion_matrix or the results table will be pivoted.
    # for modality in feature_dict:
    #     features=feature_dict[modality]
    #     # modality_name=modality ##0623
    #     X_train, X_test, y_train, y_test= train_test_split(features, my_labels, test_size=0.20,stratify=my_labels,random_state=my_random_state)
    #     # fitting
    #     svc_model = SVC(class_weight="balanced",kernel='rbf')
    #     svc_model.fit(X_train,y_train)
    #     y_pred_svc = svc_model.predict(X_test)
    #     # evaluate performance
    #     accuracy = accuracy_score(y_test, y_pred_svc)
    #     balanced_accuracy = balanced_accuracy_score(y_test, y_pred_svc)
    #     con_mat=confusion_matrix(y_pred_svc, y_test) ##0623
    #     print("Acc. using", modality, " (typical split) is:",accuracy, "and the Balanced Accuracy is",balanced_accuracy,"with the following confusion matrix:","\n", con_mat)   
    
    # ##### Cross Validation
    # scores_printing={}
    # rowIDs=["Accuracy","balanced Accuracy"]
    # for modality in feature_dict:
    #     features=feature_dict[modality]
    #     # modality_name=modality ##0623
    #     svc_model = SVC(class_weight="balanced", random_state=my_random_state, kernel='rbf')
    #     cv = StratifiedKFold(5, random_state=my_random_state,shuffle=True)
    #     scores_acc = cross_val_score(svc_model, features, my_labels, cv=cv, n_jobs=None)
    #     scores_bal_acc = cross_val_score(svc_model, features, my_labels, cv=cv, n_jobs=None,scoring="balanced_accuracy")
    #     performance_metrics=[np.round(np.mean(scores_acc),4),np.round(np.mean(scores_bal_acc),4)]
    #     scores_printing[modality]=performance_metrics
    #     most_acc_modality=max(scores_printing, key = lambda k: scores_printing[k])

    ##### Single split and Cross Validation
    my_random_state = config["default_random_state"]
    scores_printing={}
    rowIDs=["Accuracy","balanced Accuracy"]
    for modality in feature_dict:
        features=feature_dict[modality]
        # modality_name=modality ##0623
        X_train, X_test, y_train, y_test= train_test_split(features, my_labels, test_size=0.20,stratify=my_labels,random_state=my_random_state)
        # fitting
        svc_model = SVC(class_weight="balanced", kernel='rbf')
        svc_model.fit(X_train,y_train)
        y_pred_svc = svc_model.predict(X_test)
        # evaluate performance
        accuracy = accuracy_score(y_test, y_pred_svc)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred_svc)
        con_mat=confusion_matrix(y_test, y_pred_svc)  ##0623
        print("Acc. using", modality, " (typical split) is:",accuracy, "and the Balanced Accuracy is",balanced_accuracy,"with the following confusion matrix:","\n", con_mat)   

        svc_cv_model = SVC(class_weight="balanced", random_state=my_random_state, kernel='rbf')
        cv = StratifiedKFold(5, random_state=my_random_state,shuffle=True)
        scores_acc = cross_val_score(svc_cv_model, features, my_labels, cv=cv, n_jobs=None)
        scores_bal_acc = cross_val_score(svc_cv_model, features, my_labels, cv=cv, n_jobs=None,scoring="balanced_accuracy")
        performance_metrics=[np.round(np.mean(scores_acc),4),np.round(np.mean(scores_bal_acc),4)]
        scores_printing[modality]=performance_metrics
        most_acc_modality=max(scores_printing, key = lambda k: scores_printing[k])

    f.write("\nUnimodal Classification: \n")
    f.write(tabulate(scores_printing,headers="keys",showindex=rowIDs))
    f.write("\n")
    return(most_acc_modality)
     
def concatenation_classification(config,feature_dict,my_labels):
    # Getting how many features are there
    # tot_num_features=0
    # for modality in feature_dict:
    # tot_num_features=tot_num_features+len(feature_dict[modality][0])
    my_random_state = config["default_random_state"]
    tot_num_features = sum([len(feature[0]) for feature in feature_dict.values()]) #0623

    concatenated_features=np.zeros(shape=(len(my_labels),tot_num_features))
    for row in range(0,len(my_labels)):
       concatenated_features[row,:]=np.concatenate((feature_dict["EEG"][row],feature_dict["ECG"][row],feature_dict["Eye"][row],feature_dict["Seat"][row]),axis=0)  
    indices = np.arange(len(my_labels))
    
    X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(concatenated_features, my_labels, indices,test_size=0.20,stratify=my_labels,random_state=my_random_state)

    # fitting
    svc_model = SVC(class_weight="balanced",kernel='rbf')
    svc_model.fit(X_train,y_train)
    y_pred_svc = svc_model.predict(X_test)

    # evaluate performance
    accuracy = accuracy_score(y_test, y_pred_svc)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred_svc)
    con_mat=confusion_matrix(y_pred_svc, y_test)
    #print("Acc. using concatenation of features is ",accuracy, "and the Balanced Accuracy is",balanced_accuracy,"with the following confusion matrix:","\n", con_mat)  
    
    svc_cv_model= SVC(class_weight="balanced", random_state=my_random_state)
    cv = StratifiedKFold(5, random_state=my_random_state,shuffle=True)
    scores_acc = cross_val_score(svc_cv_model, concatenated_features, my_labels, cv=cv, n_jobs=None)
    scores_bal_acc = cross_val_score(svc_cv_model, concatenated_features, my_labels, cv=cv, n_jobs=None,scoring="balanced_accuracy")
    final_score=np.mean(scores_acc) 
    final_bal_score=np.mean(scores_bal_acc)
    f.write("\nConcateation Classification: \n")
    f.write("---Accuracy:%s \n" %np.round(final_score,4))
    f.write("---Balanced Accuracy:%s \n" %np.round(final_bal_score,4))
    return concatenated_features

def majority_voting_classification(config,my_labels,feature_dict,most_acc_modality,tot_num_events):
    my_random_state = config["default_random_state"]
    c = 0
    for modality in feature_dict:
        features=feature_dict[modality]
        modality_name = modality
        if c == 0:
           X_train, X_test, y_train, y_test = train_test_split(features, my_labels, test_size=0.20,stratify=my_labels,random_state=my_random_state)
           predictions_ready=np.ndarray(shape=(len(y_test),len(feature_dict)))
           final_decision=np.ndarray(shape=(len(y_test)))
        # fitting
        svc_model = SVC(class_weight="balanced",kernel='rbf')
        svc_model.fit(X_train,y_train)
        y_pred_svc = svc_model.predict(X_test)
        # evaluate performance
        accuracy = accuracy_score(y_test, y_pred_svc)
        con_mat=confusion_matrix(y_pred_svc, y_test)
        predictions_ready[:,c]=y_pred_svc
        c=c+1
    pred_count=np.ndarray(shape=(len(predictions_ready),tot_num_events))
    
    for i in range(0,len(predictions_ready)):
        my_row=predictions_ready[i][:]
        cmax=0
        for ii in range(1,tot_num_events+1):
            pred_count[i,ii-1]=np.count_nonzero(my_row ==ii)   # NoEv, ll, acc, rl, bra
        final_decision[i]=np.argmax(pred_count[i][:])   
        final_decision[i]=int(final_decision[i]+1)    
    num_correct=0
    for i in range(0,len(final_decision)):
        if  y_test[i]==final_decision[i]:
            num_correct=num_correct+1
    maj_vot_accuracy=num_correct/len(final_decision)
    #print("Acc. using Majority Voting is:", maj_vot_accuracy)


    mod_dict={ "EEG"  : 0,
                    "ECG"  : 1,
                    "Eye"  : 2,
                    "Seat" : 3
                    }

    dummy_features=feature_dict["EEG"].copy()                       
    # Just to generate the training and testing indexes for the folds
    x, y = shuffle(dummy_features, my_labels, random_state=my_random_state)

    skfolds = StratifiedKFold(n_splits=5)
    maj_vot_acc_all_folds=[]
    for fold, (train_index, test_index) in enumerate(skfolds.split(x, y)):
        my_train_index= train_index;my_test_index= test_index
        predictions_ready=np.ndarray(shape=(len(my_test_index),len(feature_dict)))
        final_decision=np.ndarray(shape=(len(my_test_index))) 
        c = 0
        for modality in feature_dict:
            features=feature_dict[modality]
            features, my_labels = shuffle(features, my_labels, random_state=my_random_state)
            X_train, X_test, y_train, y_test= np.array(features)[my_train_index], np.array(features)[my_test_index], np.array(my_labels)[my_train_index], np.array(my_labels)[my_test_index]
            # fitting
            svc_model = SVC(class_weight="balanced", random_state=my_random_state,kernel='rbf')
            svc_model.fit(X_train,y_train)
            y_pred_svc = svc_model.predict(X_test)
            predictions_ready[:,c]=y_pred_svc
            c+=1

        # Performing the actual majority voting
        num_modalities=len(list(set(my_labels)))
        pred_count=np.ndarray(shape=(len(predictions_ready),num_modalities))  # array to save the number of votes for each event
        for testing_event in range(0,len(predictions_ready)):
            my_row=predictions_ready[testing_event][:];
            previous_highest_ID=0
            current_count=[]
            event_ID=1
            tie=0
            for mod in range(0,num_modalities):  # loop over the possible modalities
                current_count=np.count_nonzero(my_row==event_ID)
                if current_count>previous_highest_ID:
                    highest_ID=event_ID
                    previous_highest_ID=current_count  
                else:
                    unique_elements, counts = np.unique(my_row, return_counts=True)
                    if np.max(counts) <= len(unique_elements) / 2:
                        tie=1
                event_ID+=1
            if tie==0:
                final_decision[testing_event]=highest_ID 
            else:
                final_decision[testing_event]=my_row[mod_dict[most_acc_modality]]

        num_correct=0
        for i in range(0,len(final_decision)):
            if  y_test[i]==final_decision[i]:
                num_correct=num_correct+1
        maj_vot_accuracy_single_fold=num_correct/len(final_decision)
        maj_vot_acc_all_folds.append(maj_vot_accuracy_single_fold)
    maj_vot_acc_final=sum(maj_vot_acc_all_folds)/5     
    f.write("\nMajority Voting Classification: \n")
    f.write("---Accuracy:%s" %np.round(maj_vot_acc_final,4))
    return 
                                    
## NEW
def partial_classification(config, orig_feature_dict, orig_labels):
    partial_classification_config = config["partial_classification_config"]
    
    for case in partial_classification_config:
        category1 = partial_classification_config[case]['category1']
        category2 = partial_classification_config[case]['category2']
        feature_dict=orig_feature_dict.copy()
        my_labels= orig_labels.copy()
        
        c=1
        for i in category1:
            if i==1:
                d=0
                while d<len(my_labels):
                    if my_labels[d]==c: # c is the event ID
                        my_labels[d]=6  # dummy_ID
                    d=d+1
            c=c+1
        c=1
        for i in category2:
            if i==1:
                d=0
                while d<len(my_labels):
                    if my_labels[d]==c:
                        my_labels[d]=7  # dummy_ID
                    d=d+1
            c=c+1            

        d_delete=[]
        d=0
        while d<len(my_labels):
            if my_labels[d]!=6 and my_labels[d]!=7:
                d_delete.append(d)
            d=d+1
        modified_my_labels=np.delete(my_labels,d_delete)
        for modality in feature_dict:
            feature_dict[modality]=np.delete(feature_dict[modality],d_delete,axis=0)
        f.write("\n######%s " %case)
        unimodal_classification(config, feature_dict, modified_my_labels)
        concatenation_classification(config,feature_dict, modified_my_labels)
        f.write("\n#####################################")
        feature_dict=orig_feature_dict.copy()
        my_labels= orig_labels.copy()
    return my_labels

def SEVF_training_features(config, training_features,training_labels):
    event_level_order = config["level_order"]["event"]
    SEVF_training_labels_level_dict={}
    SEVF_training_features_level_dict={}    # NoEv, ll, Acc, rl, Braking
    for level in range(1,4):
        if event_level_order[level-1]=="Braking":
                category1=[0,0,0,0,1]
                base_category1=category1.copy()
        elif event_level_order[level-1]=="Acceleration":
                category1=[0,0,1,0,0]
                base_category2=category1.copy()
        elif event_level_order[level-1]=="Lane-change":
                category1=[0,1,0,1,0]

        if level==1:
           category2 = [1 if element == 0 else 0 for element in category1]
        if level==3:
           category2 =[1,0,0,0,0]
        if level==2:
           category2 =[0 if any(pair) else 1 for pair in zip(base_category1, base_category2)]


        if level==1:
            orig_training_features=training_features.copy()
            orig_labels=training_labels.copy()
        else:
            training_features=orig_training_features.copy()
            training_labels= orig_labels.copy()
        
        c=1
        for i in category1:
            if i==1:
                d=0
                while d<len(training_labels):
                    if training_labels[d]==c:
                        training_labels[d]=6  # dummy_ID
                    d=d+1
            c=c+1
    
        c=1
        for i in category2:
            if i==1:
                d=0
                while d<len(training_labels):
                    if training_labels[d]==c:
                        training_labels[d]=7  # dummy_ID
                    d=d+1
            c=c+1            

        d_delete=[]
        d=0
        while d<len(training_labels):
            if training_labels[d]!=6 and training_labels[d]!=7:
                d_delete.append(d)
            d=d+1

        SEVF_training_labels_level_dict[level]=np.delete(training_labels,d_delete)
        SEVF_training_features_level_dict[level]=np.delete(training_features,d_delete,axis=0)

    return(SEVF_training_labels_level_dict,SEVF_training_features_level_dict)

def SEVF_feature_selection (SEVF_training_features_level_dict,SEVF_training_labels_level_dict):

    feature_importance_dic={}
    important_features_index_dic={}
    not_important_features_index_dic={}
    important_features_dic={}
    for level in [1,2,3]:
        ridge = RidgeCV(alphas=[.1,1,10]).fit(SEVF_training_features_level_dict[level], SEVF_training_labels_level_dict[level])
        
        importance = np.abs(ridge.coef_)
        feature_importance_dic[level]=importance
        temp_important_features_index=[]
        temp_not_important_features_index=[]
        c=0
        my_per= np.percentile(importance,20)
        for i in importance:
            if i >my_per:
                temp_important_features_index.append(c)
            else:
                temp_not_important_features_index.append(c)
            c+=1
        important_features_index_dic[level]= temp_important_features_index
        not_important_features_index_dic[level]= temp_not_important_features_index

        temp_data_dic= np.array(SEVF_training_features_level_dict[level])
        c=len(temp_data_dic[1,:])-1            # A reverse counter to go over features
        for feature in sorted(importance, reverse=True):
            if feature < my_per:
                temp_data_dic=np.delete(temp_data_dic,c,axis=1)
            c-=1
        important_features_dic[level]=temp_data_dic

    return(important_features_dic,important_features_index_dic,not_important_features_index_dic) 

def SEVF_modality_selection (SEVF_training_features_level_dict,SEVF_training_labels_level_dict,mod_level_order):
    
    important_features_index_dic={}
    not_important_features_index_dic={}
    important_features_dic={}
    for level in [1,2,3]:
        chosen_mod=mod_level_order[level-1]
        if chosen_mod=="EEG":
            important_features_dic[level]=np.array(SEVF_training_features_level_dict[level])[:,0:18]    # EEG
            important_features_index_dic[level]=list(range(0, 17))
            not_important_features_index_dic[level]=[x for x in range(64) if  x > 17]
        elif chosen_mod=="ECG":
            important_features_dic[level]=np.array(SEVF_training_features_level_dict[level])[:,18:25]    # ECG
            important_features_index_dic[level]=list(range(18, 24))
            not_important_features_index_dic[level]=[x for x in range(64) if  x<18 or x > 24]
        elif chosen_mod=="Eye":
            important_features_dic[level]=np.array(SEVF_training_features_level_dict[level])[:,25:28]    # Eye
            important_features_index_dic[level]=list(range(25, 27))
            not_important_features_index_dic[level]=[x for x in range(64) if  x<25 or  x> 27]
        elif chosen_mod=="Seat":
            important_features_dic[level]=np.array(SEVF_training_features_level_dict[level])[:,28:64]    # Seat
            important_features_index_dic[level]=list(range(28, 63))
            not_important_features_index_dic[level]=[x for x in range(64) if x < 28 or x > 63]

    return(important_features_dic,important_features_index_dic,not_important_features_index_dic) 

def SEVF_classifier_training(config,important_features_dic,SEVF_training_labels_level_dict):
    my_random_state = config["default_random_state"]
    classifier_dict={}
    for level in [1,2,3]:  
        # fitting
        #the_Weights={6:1.75,
        #             7:1}
        svc_model = SVC(class_weight="balanced", random_state=my_random_state,kernel='rbf')
        svc_model.fit(important_features_dic[level],SEVF_training_labels_level_dict[level])   
        classifier_dict[level]=svc_model
    return (classifier_dict)

def SEVF_classifier_testing(config,classifier_dict,testing_features,testing_labels,important_features_index_dic,not_important_features_index_dic):
    event_level_order=config["level_order"]["event"]
    mapping = config["mapping"]

    actual_label_1=mapping[event_level_order[0]]
    actual_label_2=mapping[event_level_order[1]]
    actual_label_3=mapping[event_level_order[2]]

    
    clf=classifier_dict[1]           
    testing_features1=np.delete(testing_features,not_important_features_index_dic[1],axis=1)
    y_pred_1 = clf.predict(testing_features1)
    predictions_1=y_pred_1

    clf=classifier_dict[2]
    testing_features2=np.delete(testing_features,not_important_features_index_dic[2],axis=1)
    y_pred_2 = clf.predict(testing_features2)
    predictions_2=y_pred_2

    clf=classifier_dict[3]
    testing_features3=np.delete(testing_features,not_important_features_index_dic[3],axis=1)
    y_pred_3 = clf.predict(testing_features3)
    predictions_3=y_pred_3

    final_predictions=np.ndarray(shape=(len(testing_labels)))

    for i in  range(0,len(final_predictions)):        # The classification scheme
        if predictions_1[i]==6:
            final_predictions[i]=actual_label_1
        else:
           if predictions_2[i]==6:
              final_predictions[i]=actual_label_2
           else:
              if predictions_3[i]==6:
                  final_predictions[i]=actual_label_3
              else:
                  final_predictions[i]=1

    num_correct=0                                     # Determining accuracy
    for i in range(0,len(final_predictions)):
        if  testing_labels[i]==final_predictions[i]:
            num_correct=num_correct+1
    SEVF_acc=num_correct/len(final_predictions)
    return(SEVF_acc)

def SEVF(config, my_labels, concatenated_features):                   
    print("SEVF has started \n")
    print("----------- \n")

    mod_level_order = config["level_order"]["modality"]
    event_level_order = config["level_order"]["event"]
    my_random_state = config["default_random_state"]

    f.write("-The Sequential events are: %s \n" %event_level_order)
    f.write("-The Sequential modalities are: %s \n" %mod_level_order)

    x, y = shuffle(concatenated_features, my_labels, random_state=my_random_state)
    skfolds = StratifiedKFold(n_splits=5)
    SEVF_acc_all=[]
    for fold, (train_index, test_index) in enumerate(skfolds.split(x, y)):
        my_train_index= train_index;my_test_index= test_index

        # Getting the data for this fold
        training_features,training_labels=np.array(x[my_train_index]),np.array(y[my_train_index])
        testing_features,testing_labels=np.array(x[my_test_index]),np.array(y[my_test_index])
        
        # Finding the features for each of the levels
        SEVF_training_labels_level_dict,SEVF_training_features_level_dict=SEVF_training_features(config,training_features,training_labels)
        
        # Finding the best features for each of the levels
        #important_features_dic,important_features_index_dic,not_important_features_index_dic=SEVF_feature_selection(SEVF_training_features_level_dict,SEVF_training_labels_level_dict)       
        important_features_dic,important_features_index_dic,not_important_features_index_dic=SEVF_modality_selection(SEVF_training_features_level_dict,SEVF_training_labels_level_dict,mod_level_order)       
        
        # Constructing the classifier        
        classifier_dict=SEVF_classifier_training(config,important_features_dic,SEVF_training_labels_level_dict)

        # SEVF testing
        SEVF_acc=SEVF_classifier_testing(config,classifier_dict,testing_features,testing_labels,important_features_index_dic,not_important_features_index_dic)
        SEVF_acc_all.append(SEVF_acc)
    
    avg_SEVF_acc=np.mean(SEVF_acc_all)
    print("SEVF is done \n")
    print("This dyad is complete \n")
    f.write("\n-The SEVF accuracy is: %s" %np.round(avg_SEVF_acc,4))
    return ()
 
def exploratory_analysis(feature_dict, my_labels): # DO IT IF TIME ALLOWS
    EEG_features_nov=[];EEG_features_ll=[];EEG_features_rl=[];EEG_features_bra=[];EEG_features_acc=[]
    ECG_features_nov=[];ECG_features_ll=[];ECG_features_rl=[];ECG_features_bra=[];ECG_features_acc=[]
    Eye_features_nov=[];Eye_features_ll=[];Eye_features_rl=[];Eye_features_bra=[];Eye_features_acc=[]
    Seat_features_nov=[];Seat_features_ll=[];Seat_features_rl=[];Seat_features_bra=[];Seat_features_acc=[]

    for index in range(0,len(my_labels)):
        if my_labels[index]==1:
            EEG_features_nov.append(feature_dict["EEG"][index])
            ECG_features_nov.append(feature_dict["ECG"][index])
            Eye_features_nov.append(feature_dict["Eye"][index])
            Seat_features_nov.append(feature_dict["Seat"][index])
        elif  my_labels[index]==2:
            EEG_features_ll.append(feature_dict["EEG"][index])
            ECG_features_ll.append(feature_dict["ECG"][index])
            Eye_features_ll.append(feature_dict["Eye"][index])
            Seat_features_ll.append(feature_dict["Seat"][index])
        elif  my_labels[index]==3:
            EEG_features_acc.append(feature_dict["EEG"][index])
            ECG_features_acc.append(feature_dict["ECG"][index])
            Eye_features_acc.append(feature_dict["Eye"][index])
            Seat_features_acc.append(feature_dict["Seat"][index])
        elif  my_labels[index]==4:
            EEG_features_rl.append(feature_dict["EEG"][index])
            ECG_features_rl.append(feature_dict["ECG"][index])
            Eye_features_rl.append(feature_dict["Eye"][index])
            Seat_features_rl.append(feature_dict["Seat"][index])
        elif  my_labels[index]==5:
            EEG_features_bra.append(feature_dict["EEG"][index])
            ECG_features_bra.append(feature_dict["ECG"][index])
            Eye_features_bra.append(feature_dict["Eye"][index])
            Seat_features_bra.append(feature_dict["Seat"][index])
    return()

###################################################################################### MAIN CODE
###################################################################################### 
###################################################################################### 
###################################################################################### 
######################################################################################

 
# Import needed APIs
import tkinter
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import csv
import mne
import math
import heartpy as hp
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs
from mne.time_frequency import psd_array_welch
from mne import Epochs
mne.set_log_level("ERROR")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit,cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.linear_model import RidgeCV
from datetime import datetime
from scipy import stats
from sys import stderr
from tabulate import tabulate
import json

config_file = open('config.json')
config = json.load(config_file)
config_file.close()

# Global inputs
# EEG_freq=config["freq"]["eeg"]
# Eye_freq=config["freq"]["eye"]
# Seat_freq=config["freq"]["seat"]
# ECG_freq=config["freq"]["ecg"]
EEG_window_size=config["window_size"]["eeg"]     # in [s]
Eye_window_size=config["window_size"]["eye"]     # in [s]
Seat_window_size=config["window_size"]["seat"]     # in [s]
ECG_window_size=config["window_size"]["ecg"]      # in [s] 

# my_random_state=42                                                 
# mod_level_order=config["level_order"]["modality"]
# event_level_order=config["level_order"]["event"]

# mapping = config["mapping"]

# actual_label_1=mapping[event_level_order[0]]
# actual_label_2=mapping[event_level_order[1]]
# actual_label_3=mapping[event_level_order[2]]

### Path

root_path = config["root_path"]

# Generating a results report
now = datetime.now()                       # Current date and time
dt = now.strftime("%d%m%Y_%H%M%S")   

f = open(root_path+r"Reports\Results_Report%s.txt" %dt, "w")
f.write("############################################################################ RESULTS REPORT ############################################################################ \n")

# Dyad-specific inputs # passenger IDs: 4069, 4207, and 1021
dyads = config["dyads"]  # Labels of considered dyads
# dyads = [1]
#f.write("-Time Stamp is:%s \n" %dt)
f.write("-Considered dyads are:%s \n" %dyads)
f.write("-Considered modalities are:EEG, ECG, Eye position, and Seat sensors \n" )
f.write("-The window sizes used for the four modalities are: %s, %s, %s, and %s \n" %(EEG_window_size,ECG_window_size,Eye_window_size,Seat_window_size ))
f.write("-The events considered are: Braking, Acceleration, Left-turn, and Right-turn, added to baseline events \n")

# # Master loop over dyads                          
# def new_func(Seat_window_size):
# return Seat_window_size

for d in dyads:  
    print("################################################ DYAD:",d, " ################################################")
    f.write("\n###################################################################################### Dyad: %s \n" %d)

    # Import event file and data files
    root = Tk()
    root.withdraw()
    tsvfile =  open(root_path+r"D%s\D%s_event_file.tsv" %(d,d), 'r')                                                                                   # Open the event file
    temp_EEG_data = pd.read_csv(root_path+r"D%s\D%s_eeg_data.csv" %(d,d),skiprows=0, usecols=[*range(1, 21)])
    EEG_data=temp_EEG_data.transpose() # Open the EEG file
    
    Eye_data = pd.read_csv(root_path+r"D%s\D%s_eye_data.csv" %(d,d),skiprows=0)                                                                        # Open the Eye file
    
    Seat_data = pd.DataFrame(pd.read_csv(root_path+r"D%s\D%s_seats_data.csv" %(d,d),skiprows=0))
    Seat_data.drop(['Time UTC (YYYYMMDD_HHMMSS.mmm)','Time Arduino (Micros since start)'], axis = 1, inplace = True) # Open the Seat file
    
    temp_ECG_data = pd.read_csv(root_path+r"D%s\D%s_ecg_data.csv" %(d,d),skiprows=0)
    ECG_data =temp_ECG_data[:].my_data                               # Open the ECG file
    print("Files are imported","\n","-----------")

    data_files_dict = {
        "eeg":EEG_data,
        "ecg":ECG_data,
        "eye":Eye_data,
        "seat":Seat_data
    }       
    # Event_extraction and generation of baseline (no-event) events
    event_dict, counts_dict, tot_num_events = event_extraction(config,tsvfile,d)

    # Data synchronization
    event_indexes_dict,Seat_video_synch_diff,baseline_start_time,baseline_end_time=modality_synchronization(config,dyads,d,event_dict)
    print("Modalities are synchronized","\n","-----------")

    # Preprocessing and feature extraction
    feature_dict, my_labels = feature_extraction(config,data_files_dict,d,event_dict,event_indexes_dict,tot_num_events,Seat_video_synch_diff,baseline_start_time,baseline_end_time)
    print("All features are extracted")

    # Classification
    f.write("ALL EVENTS \n")
    most_acc_modality = unimodal_classification(config,feature_dict,my_labels)
    concatenated_features = concatenation_classification(config,feature_dict,my_labels)
    majority_voting_classification(config,my_labels,feature_dict,most_acc_modality,tot_num_events)

    f.write("\n#####################################PARTIAL CLASSIFICATION: \n")
    my_labels=partial_classification(config, feature_dict, my_labels)       

    f.write("SEVF: \n")
    # Sequential Event-Based Fusion (SEVF)
    SEVF(config, my_labels, concatenated_features)

    # Exploratory_analysis
    exploratory_analysis(feature_dict, my_labels)
    f.close()
    break

# Finish writing the results report
f.close()