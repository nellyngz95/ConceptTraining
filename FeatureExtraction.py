import essentia.standard as ess
import essentia
import os
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf

#Read each label of the soundfile so we can add it to the .csv afterwards.

def extract_labels(path):
    # Initialize a dictionary to hold the labels for each file
    label_data = {
        'Label': []  # List to store each label found
    }

    # Traverse the directory to collect labels based on the filename
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.wav'):
                try:
                    # Assuming labels are the second part of the filename (e.g., file-label.wav)
                    label = filename.split('-')[1].split('.')[0]
                    label_data['Label'].append(label)
                except IndexError:
                    print(f"Filename format issue with: {filename}")
                    label_data['Label'].append(None)  # Assign None if label extraction fails

    # Convert the dictionary to a DataFrame
    return pd.DataFrame(label_data)

#Functions for extracting features
#1. Spectral features

def extract_spectral_features(path):
    data = {
        'Frequency1': [], 'Amplitude1': [],
        'Frequency2': [], 'Amplitude2': [],
        'Frequency3': [], 'Amplitude3': [],
        'Frequency4': [], 'Amplitude4': [],
        'Frequency5': [], 'Amplitude5': []
    }

    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)

                try:
                    loader = ess.MonoLoader(filename=file_path, sampleRate=44100)
                    audio = loader()

                    frameSize = 2048
                    hopSize = 1024
                    windowing = ess.Windowing(type='blackmanharris62', zeroPadding=2048)
                    spectrum = ess.Spectrum(size=frameSize)
                    spectral_peaks = ess.SpectralPeaks(maxPeaks=5)

                    all_frequencies = []
                    all_amplitudes = []

                    for frame in ess.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
                        frame_spectrum = spectrum(windowing(frame))
                        peaks_frame = spectral_peaks(frame_spectrum)
                        all_frequencies.extend(peaks_frame[0])
                        all_amplitudes.extend(peaks_frame[1])

                    # Get the top 5 spectral peaks
                    peaks = sorted(zip(all_frequencies, all_amplitudes), key=lambda x: x[1], reverse=True)[:5]
                    frequencies, amplitudes = zip(*peaks) if peaks else ([], [])

                    # Pad with zeros if fewer than 5 peaks
                    frequencies = list(frequencies) + [0] * (5 - len(frequencies))
                    amplitudes = list(amplitudes) + [0] * (5 - len(amplitudes))

                    # Append values to the data dictionary
                    for i in range(5):
                        data[f'Frequency{i+1}'].append(frequencies[i])
                        data[f'Amplitude{i+1}'].append(amplitudes[i])

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    # Append NaN values if an error occurs
                    for i in range(5):
                        data[f'Frequency{i+1}'].append(float('nan'))
                        data[f'Amplitude{i+1}'].append(float('nan'))

    return pd.DataFrame(data)
    
# 2. Extract temporal features

def extract_tempfeatures(path):
    # Initialize lists to collect individual temporal features for all files
    loudness_data = []
    rms_data = []
    spectral_flux_data = []
    centroid_data = []
    high_freq_content_data = []
    zcr_data = []
    energy_data = []
    pitch_salience_data = []
    effective_duration_data = []
    decrease_data = []
    intensity_data = []
    dyn_complexity_data = []
    ldb_data = []
    cm1_data = []
    cm2_data = []
    cm3_data = []
    cm4_data = []
    cm5_data = []

    # Traverse the directory to process each .wav file
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)

                # Initialize default feature values for the current file as NaN
                features = {
                    'Loudness': np.nan,
                    'RMS': np.nan,
                    'SpectralFlux': np.nan,
                    'Centroid': np.nan,
                    'HighFrequencyContent': np.nan,
                    'ZCR': np.nan,
                    'Energy': np.nan,
                    'PitchSalience': np.nan,
                    'EffectiveDuration': np.nan,
                    'Decrease': np.nan,
                    'Intensity': np.nan,
                    'DynComplexity': np.nan,
                    'LDB': np.nan,
                    'CM1': np.nan,
                    'CM2': np.nan,
                    'CM3': np.nan,
                    'CM4': np.nan,
                    'CM5': np.nan
                }

              
                    # Load the audio file
                loader = ess.MonoLoader(filename=file_path, sampleRate=44100)
                audio = loader()
                features['Loudness'] = ess.Loudness()(audio)
                features['RMS'] = ess.RMS()(audio)
                features['SpectralFlux'] = ess.Flux()(audio)
                features['Centroid'] = ess.Centroid()(audio)
                features['HighFrequencyContent'] = ess.HFC()(audio)
                features['ZCR'] = ess.ZeroCrossingRate()(audio)
                features['Energy'] = ess.Energy()(audio)
                features['PitchSalience'] = ess.PitchSalience()(audio)
                features['EffectiveDuration'] = ess.EffectiveDuration()(audio)
                features['Decrease'] = ess.Decrease()(audio)
                features['Intensity'] = ess.Intensity()(audio)

                    # Dynamic complexity and central moments (handling tuple unpacking)
                dyncomp = ess.DynamicComplexity()(audio)
                features['DynComplexity'], features['LDB'] = dyncomp  # Unpacking if it returns a tuple
                CM = ess.CentralMoments()(audio)
                features['CM1'], features['CM2'], features['CM3'], features['CM4'], features['CM5'] = CM

            

                # Append each feature individually to the respective lists
                loudness_data.append(features['Loudness'])
                rms_data.append(features['RMS'])
                spectral_flux_data.append(features['SpectralFlux'])
                centroid_data.append(features['Centroid'])
                high_freq_content_data.append(features['HighFrequencyContent'])
                zcr_data.append(features['ZCR'])
                energy_data.append(features['Energy'])
                pitch_salience_data.append(features['PitchSalience'])
                effective_duration_data.append(features['EffectiveDuration'])
                decrease_data.append(features['Decrease'])
                intensity_data.append(features['Intensity'])
                dyn_complexity_data.append(features['DynComplexity'])
                ldb_data.append(features['LDB'])
                cm1_data.append(features['CM1'])
                cm2_data.append(features['CM2'])
                cm3_data.append(features['CM3'])
                cm4_data.append(features['CM4'])
                cm5_data.append(features['CM5'])

    # Create a DataFrame from all the lists of features
    temp_features_df = pd.DataFrame({
        'Loudness': loudness_data,
        'RMS': rms_data,
        'SpectralFlux': spectral_flux_data,
        'Centroid': centroid_data,
        'HighFrequencyContent': high_freq_content_data,
        'ZCR': zcr_data,
        'Energy': energy_data,
        'PitchSalience': pitch_salience_data,
        'EffectiveDuration': effective_duration_data,
        'Decrease': decrease_data,
        'Intensity': intensity_data,
        'DynComplexity': dyn_complexity_data,
        'LDB': ldb_data,
        'CM1': cm1_data,
        'CM2': cm2_data,
        'CM3': cm3_data,
        'CM4': cm4_data,
        'CM5': cm5_data
    })

    return temp_features_df

#3. Statistical features
def stat_features(path):
    # Initialize lists to collect statistical features for all files
    mean_data = []
    median_data = []
    variance_data = []
    instant_power_data = []
    crest_data = []
    max_to_total_data = []
    min_to_total_data = []
    tc_to_total_data = []
    flatness_sfx_data = []
    log_attack_time_data = []
    attack_start_data = []
    attack_stop_data = []
    spread_data = []
    skewness_data = []
    kurtosis_data = []

    # Traverse directory to process each .wav file
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)

                # Initialize default feature values for the current file
                stats = {
                    'Mean': 0,
                    'Median': 0,
                    'Variance': 0,
                    'InstantPower': 0,
                    'Crest': 0,
                    'MaxToTotal': 0,
                    'MinToTotal': 0,
                    'TCToTotal': 0,
                    'FlatnessSFX': 0,
                    'LogAttackTime': 0,
                    'AttackStart': 0,
                    'AttackStop': 0,
                    'Spread': 0,
                    'Skewness': 0,
                    'Kurtosis': 0
                }

                # Load audio
                loader = ess.MonoLoader(filename=file_path, sampleRate=44100)
                audio = loader()

                # Extract features
                stats['Mean'] = ess.Mean()(audio)
                stats['Median'] = ess.Median()(audio)
                stats['Variance'] = ess.Variance()(audio)
                stats['InstantPower'] = ess.InstantPower()(audio)
                stats['Crest'] = ess.Crest()(abs(audio))
                
                # Distribution and temporal features
                CM = ess.CentralMoments()(audio)
                DS = ess.DistributionShape()(CM)
                stats['Spread'], stats['Skewness'], stats['Kurtosis'] = DS

                envelope = ess.Envelope()(audio)
                stats['TCToTotal'] = ess.TCToTotal()(envelope)
                stats['FlatnessSFX'] = ess.FlatnessSFX()(envelope)
                stats['MaxToTotal'] = ess.MaxToTotal()(envelope)
                stats['MinToTotal'] = ess.MinToTotal()(envelope)

                # Attack features
                ltt, lst, lstop = ess.LogAttackTime()(envelope)
                stats['LogAttackTime'] = ltt
                stats['AttackStart'] = lst
                stats['AttackStop'] = lstop

                # Append each feature individually to the respective lists
                mean_data.append(stats['Mean'])
                median_data.append(stats['Median'])
                variance_data.append(stats['Variance'])
                instant_power_data.append(stats['InstantPower'])
                crest_data.append(stats['Crest'])
                max_to_total_data.append(stats['MaxToTotal'])
                min_to_total_data.append(stats['MinToTotal'])
                tc_to_total_data.append(stats['TCToTotal'])
                flatness_sfx_data.append(stats['FlatnessSFX'])
                log_attack_time_data.append(stats['LogAttackTime'])
                attack_start_data.append(stats['AttackStart'])
                attack_stop_data.append(stats['AttackStop'])
                spread_data.append(stats['Spread'])
                skewness_data.append(stats['Skewness'])
                kurtosis_data.append(stats['Kurtosis'])

    # Create a DataFrame from all the lists of features
    stats_features_df = pd.DataFrame({
        'Mean': mean_data,
        'Median': median_data,
        'Variance': variance_data,
        'InstantPower': instant_power_data,
        'Crest': crest_data,
        'MaxToTotal': max_to_total_data,
        'MinToTotal': min_to_total_data,
        'TCToTotal': tc_to_total_data,
        'FlatnessSFX': flatness_sfx_data,
        'LogAttackTime': log_attack_time_data,
        'AttackStart': attack_start_data,
        'AttackStop': attack_stop_data,
        'Spread': spread_data,
        'Skewness': skewness_data,
        'Kurtosis': kurtosis_data
    })

    return stats_features_df
    
# 4. Timbral Features
def timbre_features(path):
    # Initialize lists to collect timbre features for all files
    pitch_salience_data = []
    pitch_values_data = []
    pitch_confidence_data = []

    # Traverse directory to process each .wav file
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)

                # Load audio
                loader = ess.MonoLoader(filename=file_path, sampleRate=44100)
                audio = loader()

                # Initialize pitch extractor and values
                try:
                    # Initialize pitch extractor
                    pitch_extractor = ess.PredominantPitchMelodia(frameSize=2048, hopSize=128)
                    
                    # Extract pitch values and pitch confidence
                    pitch_values, pitch_confidence = pitch_extractor(audio)
                    
                    # Calculate PitchSalience
                    pitch_salience = ess.PitchSalience()(audio)

                    # Compute median of pitch_values and pitch_confidence
                    pitch_values_median = np.median(pitch_values) if len(pitch_values) > 0 else 0
                    pitch_confidence_median = np.median(pitch_confidence) if len(pitch_confidence) > 0 else 0

                    # Append the extracted features to the respective lists
                    pitch_salience_data.append(pitch_salience)
                    pitch_values_data.append(pitch_values_median)
                    pitch_confidence_data.append(pitch_confidence_median)

                except Exception as e:
                    # Append NaN values if an error occurs
                    pitch_salience_data.append(np.nan)
                    pitch_values_data.append(np.nan)
                    pitch_confidence_data.append(np.nan)

    # Create a DataFrame from the lists of timbre features
    timbre_features_df = pd.DataFrame({
        'PitchSalience': pitch_salience_data,
        'PitchValues': pitch_values_data,
        'PitchConfidence': pitch_confidence_data
    })

    return timbre_features_df
    
# 5. MFCC'S the first 13 coefficients.
def mfcs(path):
    # Initialize lists to collect MFCC features for all files
    mfcc_features_data = [[] for _ in range(13)]  # List for each of the 13 MFCC coefficients

    # Traverse directory to process each .wav file
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)

                # Load audio
                loader = ess.MonoLoader(filename=file_path, sampleRate=44100)
                audio = loader()

                # Parameters for frame-based processing
                frame_size = 2048
                hop_size = 512
                windowing = ess.Windowing(type='hann')
                spectrum = ess.Spectrum()
                mfcc_extractor = ess.MFCC()

                mfcc_list = []

                # Process audio frame-by-frame
                for frame in ess.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
                    windowed_frame = windowing(frame)
                    spectrum_frame = spectrum(windowed_frame)
                    mfcc_bands, mfcc_coefficients = mfcc_extractor(spectrum_frame)
                    mfcc_list.append(mfcc_coefficients)

                # Calculate mean MFCCs over all frames
                mfcc_means = [np.mean(coef) for coef in zip(*mfcc_list)]

                # Append each MFCC feature to its corresponding list
                for i, mfcc_mean in enumerate(mfcc_means):
                    mfcc_features_data[i].append(mfcc_mean)

    # Create a DataFrame from the lists of MFCC features
    mfcc_features_df = pd.DataFrame({
        f'MFCC_{i+1}': mfcc_features_data[i] for i in range(13)
    })

    return mfcc_features_df

#6. STFT'S
def stft(path):
    # Initialize lists to collect STFT features for all files
    stft_features_data = [[], [], []]  # Lists for SpecComplexity, RollOff, and StrongPeak

    # Traverse directory to process each .wav file
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                # Load audio
                loader = ess.MonoLoader(filename=file_path, sampleRate=44100)
                audio = loader()

                # Calculate spectrum and spectral features
                spectrum = ess.Spectrum()(audio)
                spectrum_vector = np.abs(spectrum)

                # Extract spectral features
                spec_complexity = ess.SpectralComplexity()(spectrum_vector)
                rolloff = ess.RollOff()(spectrum_vector)
                strong_peak = ess.StrongPeak()(spectrum_vector)

                # Append extracted features to their respective lists
                stft_features_data[0].append(spec_complexity)  # SpecComplexity
                stft_features_data[1].append(rolloff)  # RollOff
                stft_features_data[2].append(strong_peak)  # StrongPeak

    # Create a DataFrame from the lists of STFT features
    stft_features_df = pd.DataFrame({
        'SpecComplexity': stft_features_data[0],
        'RollOff': stft_features_data[1],
        'StrongPeak': stft_features_data[2]
    })

    return stft_features_df

#7. Function to call all the other fucntions.
def extract_all_features(path):
    # Label extraction (assuming `extract_labels` is defined)
    labels = extract_labels(path)
    
    # Spectral features (assuming `extract_spectral_features` is defined)
    
    spectral_features= extract_spectral_features(path)
    temp_features=extract_tempfeatures(path)
    stats_features=stat_features(path)
    timbres_features=timbre_features(path)
    mfc_coef=mfcs(path)
    stfts=stft(path)

   
    
    # Combine all features into a final DataFrame
    final_df = pd.concat([spectral_features,temp_features,stats_features,timbres_features,mfc_coef,stfts], axis=1)
    
    # Add labels column
    final_df['Label'] = labels
    
    return final_df

# MAIN
path = "/Users/nellygarcia/Downloads/Test" #<---- CHANGE THE PATH TO THE FOLDER THAT CONTAIN YOUR .WAV FILES
labels = extract_labels(path)
final_df = extract_all_features(path)
csv_output_path = '/Users/nellygarcia/Desktop/test_features.csv' #<---- CHANGE THE PATH TO WERE YOU WANT TO SAVE THE .CSV

# Save the DataFrame to a CSV file
final_df.to_csv(csv_output_path, index=False)

print(f"Data saved to {csv_output_path}") #<---- OPTIONAL:  PRINT THE OUTPUT 
print (final_df)
