# 2026TxstREUProject4

Project Activity 4 – Using AI and IoT to Build Smart Homes for Individuals with Autism Spectrum Disorder  (led by Dr. Liu & Dr. Carvalho)

Autism Spectrum Disorder (ASD) is a neurodevelopmental disorder characterized by difficulties in social interaction, communication, and repetitive behaviors. Individuals with ASD often face challenges in adapting to their living environment, including problems managing sensory stimuli and maintaining routines. The project aims to develop IoT and AI-driven smart home systems for individuals with ASD. These smart home systems will utilize advanced IoT and AI algorithms to create personalized environments that optimize comfort, safety, health, and well-being for individuals with ASD.

Machine learning models will analyze channel state information (CSI) data obtained from human experiments with the ESP32 microcontrollers to classify which movements are associated with ASD. 

<<<<<<< HEAD
This is the Zotero associated with this reasearch project: https://www.zotero.org/groups/6584451/txst-reu-project-/items/GRR6VSXJ/item-list.
=======
Download the input data from http://researchdata.cab.unipd.it/id/eprint/624 and unzip the file.
For your convenience, you can use the ```input_files``` inside this project folder to place the files but the scripts work whatever is the source folder.

The dataset contains Wi-Fi channel frequency response (CFR) data collected in an IEEE 802.11ac network through [NEXMON CSI](https://github.com/seemoo-lab/nexmon_csi). 
The information is collected by a monitor node (ASUS RT-AC86U router) while two terminals are exchanging traffic in channel 42 (5.21 GHz for the center frequency and 80 MHz of bandwidth) and a person acts as an obstacle for the transmission by performing different activities. 
The considered movements are the following: walking (W) or running (R) around, jumping (J) in place, sitting (L) or standing (S) somewhere in the room, sitting down and standing up (C) continuously, and doing arm gym (H).
The CFR data for the empty room (E) is also provided. We obtained data from three volunteers, a male, and two females.
The complete description of the dataset can be found in the reference paper.

The code for SHARP is implemented in Python and can be found in the ```Python_code``` folder inside this repository. The scripts to perform the processing are described in the following, together with the specific parameters.

### Phase sanitization
The following three scripts encode the phase sanitization algorithm detailed in Section 3.1 of the referred article.
```bash
python CSI_phase_sanitization_signal_preprocessing.py <'directory of the input data'> <'process all the files in subdirectories (1) or not (0)'> <'name of the file to process (only if 0 in the previous field)'> <'number of spatial streams'> <'number of cores'> <'index where to start the processing for each stream'> 
```
e.g., python CSI_phase_sanitization_signal_preprocessing.py ../input_files/S1a/ 1 - 1 4 0

Code Summary:
1) Loads raw .mat file with captured CSI data 
2) Reorders subcarriers negative to positive frequencies
3) Remove subcarriers with 0 and null subcarriers (these are at specific indices)
4) Separate spatial streams
5) Sign flip upper subcarriers (correction for the specific NIC used in SHARP)
6) Divide each packet's CSI by the mean amplitude to normalize because things like transmitter location shifts the amplitude on all of the subcarriers by the same amount

```bash
python CSI_phase_sanitization_H_estimation.py <'directory of the input data'> <'process all the files in subdirectories (1) or not (0)'> <'name of the file to process (only if 0 in the previous field)'> <'number of spatial streams'> <'number of cores'> <'index where to start the processing for each stream'> <'index where to stop the processing for each stream'> 
```
e.g., python CSI_phase_sanitization_H_estimation.py ../input_files/S1a/ 0 S1a_E 1 4 0 -1

Code Summary:
1) Sets up the frequency domain <-- CSI data for each subchannel
2) Builds a T-matrix to map the time-delay to the frequency domain <-- possible delays at subchannel k for all paths
3) For each spatial stream find the strongest path using the Lasso regression
4) Save the sparse impulse response and the sanitized frequency-domain channel matrix

```bash
python CSI_phase_sanitization_signal_reconstruction.py <'directory of the processed data'> <'directory to save the reconstructed data'> <'number of spatial streams'> <'number of cores'> <'index where to start the processing for each stream'> <'index where to stop the processing for each stream'> 
```
e.g., python CSI_phase_sanitization_signal_reconstruction.py ./phase_processing/ ./processed_phase/ 1 4 0 -1

Code Summary:
1) Removes packets from the start and end to get rid of transients (emitted from a WiFi device)
2) Extract amplitude and extract phase
3) Cleans phase by removing abrupt differences between subcarriers and doing least-squares phase removal

### Doppler computation
The following script computes the Doppler spectrum as described in Section 3.2 of the referred article.

```bash
python CSI_doppler_computation.py <'directory of the reconstructed data'> <'sub-directories of data'> <'directory to save the Doppler data'> <'starting index to process data'> <'end index to process data (samples from the end)'> <'number of packets in a sample'> <'number of packets for sliding operations'> <'noise level'> <--bandwidth 'bandwidth'>
```
e.g., python CSI_doppler_computation.py ./processed_phase/ S1a,S1b,S1c,S2a,S2b,S3a,S4a,S4b,S5a,S6a,S6b,S7a ./doppler_traces/ 800 800 31 1 -1.2

Code Summary:
1) Recombines phase and amplitude
2) Selects a sub-band so that only the subcarriers corresponding to that band are used (optional)
3) Multiplies each sliding window by the Haan function to prevent energy from oone velocity bin bleeding into adjacent bins
4) Converts each sliding window into Doppler frequency channel 
5) Reorders the output so that the zero velocity (no motion) is in the center
6) Calculate the power at each subcarrier 
7) Normalize (values from 0 to 1) and any value below a threshold gets replaced by the threshold value

To plot the Doppler traces use (first to plot all the antennas, second single antenna for all the activities) 
```bash
python CSI_doppler_plots_antennas.py <'directory of the reconstructed data'> <'sub-directory of data'> <'length along the feature dimension (height)'> <'sliding length'> <'labels of the activities to be considered'> <'last index to plot'>
```
e.g., python CSI_doppler_plots_antennas.py ./doppler_traces/ S7a 100 1 E,L1,W,R,J1 20000

Code Summary:
1) Look for the label abbreviations in the file name to match each file to a label
2) For the Doppler data replace any value below the threshold with the threshold and convert to dB
2) Show the Doppler spectrom for each activity for each antenna

```bash
python CSI_doppler_plots_activities.py <'directory of the reconstructed data'> <'sub-directory of data'> <'length along the feature dimension (height)'> <'sliding length'> <'labels of the activities to be considered'> <'first index to plot'> <'last index to plot'>
```
e.g., python CSI_doppler_plots_activities.py ./doppler_traces/ S7a 100 1 E,L1,W,R,J1 570 1070

Code Summary:
*Similar to above but you look at multiple activities side by side 

#### Pre-computed Doppler traces
If you want to skip the above processing steps, you can find the Doppler traces [in this Google Drive folder](https://drive.google.com/drive/folders/1SilO6VD73Lz8sjZ-KQgFnQ2IKRvggqPg?usp=sharing). In the same folder, the sanitized channel measurements for S2a and S7a are uploaded as examples in ```processed_phase```. Exaples of plots of the Doppler traces are also included.

### Dataset creation
- Create the datasets for training and validation
```bash
python CSI_doppler_create_dataset_train.py <'directory of the Doppler data'> <'sub-directories, comma-separated'> <'number of packets in a sample'> <'number of packets for sliding operations'> <'number of samples per window'> <'number of samples for window sliding'> <'labels of the activities to be considered'> <'number of streams * number of antennas'>
```
  e.g., python CSI_doppler_create_dataset_train.py ./doppler_traces/ S1a,S1b,S1c 31 1 340 30 E,L,W,R,J 4

Code Summary:
1) Sort files by streams x antennas to form multi-antenna stacks from the same recording
2) Remove static background from each Doppler trace
3) 60% train, 20% validation, 20% testing
4) Apply sliding window 

- Create the datasets for test
```bash
python CSI_doppler_create_dataset_test.py <'directory of the Doppler data'> <'sub-directories, comma-separated'> <'number of packets in a sample'> <'number of packets for sliding operations'> <'number of samples per window'> <'number of samples for window sliding'> <'labels of the activities to be considered'> <'number of streams * number of antennas'>
```
  e.g., python CSI_doppler_create_dataset_test.py ./doppler_traces/ S2a,S2b,S3a,S4a,S4b,S5a,S6a,S6b,S7a 31 1 340 30 E,L,W,R,J 4

Code Summary:
*Similar to above

### Train the learning algorithm for HAR
```bash
python CSI_network.py <'directory of the datasets'> <'sub-directories, comma-separated'> <'length along the feature dimension (height)'> <'length along the time dimension (width)'> <'number of channels'> <'number of samples in a batch'> <'name prefix for the files'> <'activities to be considered, comma-separated'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'> 
```
e.g., python CSI_network.py ./doppler_traces/ S1a 100 340 1 32 4 single_ant E,L,W,R,J

Code Summary:
1) Split each multi-antenna sample into single antenna samples
2) Build the CNN (Inception-ResNet) that outputs a probability score of each of the 5 activities
3) Train using the Adam optimizer 
4) Evalute per-antenna then merge the antenna desiciosn 
5) Run with different combinations of antennas (ie. 1,2 or 1 or 1,3,4)


### Use the trained algorithm for inference
- Run the algorithm with the test data 
```bash
python CSI_network_test.py <'directory of the datasets'> <'sub-directories, comma-separated'> <'length along the feature dimension (height)'> <'length along the time dimension (width)'> <'number of channels'> <'number of samples in a batch'> <'name prefix for the files'> <'activities to be considered, comma-separated'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'> 
```
  e.g., python CSI_network_test.py ./doppler_traces/ S7a 100 340 1 32 4 single_ant E,L,W,R,J

Code Summary:
*Similar to the structure of validation above 

- Compute the performance metrics using the output file of the test
```bash
python CSI_network_metrics.py <'name of the output file containing the metrics'> <'activities to be considered, comma-separated'>
```
  e.g., python CSI_network_metrics.py complete_different_E,L,W,R,J_S7a_band_80_subband_1 E,L,W,R,J 

Code Summary:
1) Print single antenna accuracy, precision, recall, F-score per activity
2) Print combined antenna accuracy, F-score per activity
3) Print the results from the differnt combinations of antennas

- Plot the performance metrics
```bash
python CSI_network_metrics_plot.py <'sub-directories, comma-separated'>
```
  e.g., python CSI_network_metrics_plot.py complete_different_E,L,W,R,J_S7a_band_80_subband_1 E,L,W,R,J

Code Summary:
1) Makes confusion matrix for the single-antenna predictions and then also for the combined antenna predictions

Some examples of confusion matrices can be found [in this Google Drive folder](https://drive.google.com/drive/folders/1SilO6VD73Lz8sjZ-KQgFnQ2IKRvggqPg?usp=sharing).

### Parameters
The results of the article are obtained with the parameters reported in the examples. For convenience, the repository also contains two pre-trained networks, i.e., ``single_ant_E,L,W,R,J_network.h5`` and ``single_ant_E,L,W,R,J_C_H_S_network.h5`` respectively for 5-classes and 8-classes classification problems.

### Python and relevant libraries version
Python >= 3.7.7  
TensorFlow >= 2.6.0  
Numpy >= 1.19.5  
Scipy = 1.4.1  
Scikit-learn = 0.23.2  
OSQP >= 0.6.1

## Contact
Francesca Meneghello
meneghello@dei.unipd.it
github.com/francescamen
>>>>>>> 160b4ba (trying to push modified sharp code)
