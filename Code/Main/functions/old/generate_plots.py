def plot_paradigm_timings(events_spikes, event_id, settings, params):
    import mne
    import os, sys
    import matplotlib.pyplot as plt
    fname = 'paradigm_events_' + settings.hospital + '_' + settings.patient + '.png'
    fig_paradigm = mne.viz.plot_events(events_spikes, params.sfreq_spikes, 0, event_id=event_id, show=False)
    # os.makedirs(os.path.join(settings.path2figures, settings.patient, 'misc'))
    plt.savefig(os.path.join(settings.path2figures, settings.patient, 'misc', fname))
    plt.close(fig_paradigm)