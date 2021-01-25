def get_comparison_stats(epochs1, epochs2, **kwargs):
    '''
    Run statistical tests to compare between two conditions.
    Stat results are calculated per each channel.
        
    Parameters
    ----------
    epochs1: epochs object for condition 1
    
    epochs2: epochs object for condition 2
    
    Returns
    -----
    responsive_channels: list of dicts (len of list is number of channels in epochs)
        each dict has keys - ch_name, ch_IX, T_obs, clusters, cluster_p_values, H0
    '''
    from mne.stats import permutation_cluster_test
    
    # Check that both epochs have the exact same channels, in the same order
    assert epochs1.ch_names == epochs2.ch_names, "The two epochs must contain the same channels, in the same order"

    # Collect stats from all channels
    responsive_channels = []
    for i_ch, ch_name in enumerate(epochs1.ch_names): # loop over channels
        print(f'Channel {i_ch} ({ch_name}):')
        
        # Take a single channel to get a 2D array
        condition1 = epochs1.get_data()[:, i_ch, :] 
        condition2 = epochs2.get_data()[:, i_ch, :]
        
        # Compute statistic
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([condition1, condition2], **kwargs)
        
        responsive_channels.append({'ch_name':ch_name, 'ch_IX':i_ch,
                                    'T_obs':T_obs, 'clusters':clusters, 
                                    'cluster_p_values':cluster_p_values, 'H0':H0})

    return responsive_channels
