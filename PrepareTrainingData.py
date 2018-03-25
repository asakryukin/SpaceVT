import pickle as pkl
import urllib2
import numpy as np

dst_values = []

#store all DST values from 1963 to 2010 in one array
with open('data/dst_63_10.txt', 'r') as file:
    for line in file:
        data = line[20:116]
        for i in xrange(0, 24):
            dst_values.append(float(data[i * 4:(i + 1) * 4]))

print(len(dst_values))


imf_url = 'https://cdaweb.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_'
imf_values = []

# Do the same for Imf values
for i in xrange(1963, 2011):
    #Download historical values for each year
    response = urllib2.urlopen(imf_url + str(i) + '.dat')
    content = response.read()
    data = str(content).splitlines()

    for line in data:
        splitted = line.split()
        #use only Bz value from GSE coordinate system
        imf_values.append(float(splitted[14]))

print(len(imf_values))

#Calculate meand and std values for the normalization
mean_dst = np.mean(dst_values)
mean_imf = np.mean(imf_values)

std_dst = np.std(dst_values)
std_imf = np.std(imf_values)

#Dump this data into pickles for future use
with open('data/data.pickle', 'wb') as handle:
    pkl.dump(mean_dst, handle, protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(mean_imf, handle, protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(std_dst, handle, protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(std_imf, handle, protocol=pkl.HIGHEST_PROTOCOL)


#Create training inputs, where input is data from 5 timesteps before (both DST and IMF)
dataset = []
labels = []

with open('data/training_data.csv', 'w') as file:
    for i in xrange(5, 420760):
        #Normalize data before storing
        file.write(str((dst_values[i - 5] - mean_dst) / std_dst) + ',' +
                   str((dst_values[i - 4] - mean_dst) / std_dst) + ',' +
                   str((dst_values[i - 3] - mean_dst) / std_dst) + ',' +
                   str((dst_values[i - 2] - mean_dst) / std_dst) + ',' +
                   str((dst_values[i - 1] - mean_dst) / std_dst) + ',' +
                   str((imf_values[i - 5] - mean_imf) / std_imf) + ',' +
                   str((imf_values[i - 4] - mean_imf) / std_imf) + ',' +
                   str((imf_values[i - 3] - mean_imf) / std_imf) + ',' +
                   str((imf_values[i - 2] - mean_imf) / std_imf) + ',' +
                   str((imf_values[i - 1] - mean_imf) / std_imf) + '\n'
                   )
        dataset.append([((dst_values[i - 5] - mean_dst) / std_dst),
                        ((dst_values[i - 4] - mean_dst) / std_dst),
                        ((dst_values[i - 3] - mean_dst) / std_dst),
                        ((dst_values[i - 2] - mean_dst) / std_dst),
                        ((dst_values[i - 1] - mean_dst) / std_dst),
                        ((imf_values[i - 5] - mean_imf) / std_imf),
                        ((imf_values[i - 4] - mean_imf) / std_imf),
                        ((imf_values[i - 3] - mean_imf) / std_imf),
                        ((imf_values[i - 2] - mean_imf) / std_imf),
                        ((imf_values[i - 1] - mean_imf) / std_imf)
        ])


#And labels are 5 DST hours ahead
with open('data/training_labels.csv', 'w') as file:
    for i in xrange(5, 420760):
        file.write(str((dst_values[i] - mean_dst) / std_dst) + ',' +
                   str((dst_values[i + 1] - mean_dst) / std_dst) + ',' +
                   str((dst_values[i + 2] - mean_dst) / std_dst) + ',' +
                   str((dst_values[i + 3] - mean_dst) / std_dst) + ',' +
                   str((dst_values[i + 4] - mean_dst) / std_dst) + '\n'
                   )
        labels.append([
            ((dst_values[i] - mean_dst) / std_dst),
            ((dst_values[i+1] - mean_dst) / std_dst),
            ((dst_values[i+2] - mean_dst) / std_dst),
            ((dst_values[i+3] - mean_dst) / std_dst),
            ((dst_values[i+4] - mean_dst) / std_dst)

        ])

#Split data into training and testing and dump into a pickle file
print(len(labels))
with open('data/training_data.pickle', 'wb') as handle:
    pkl.dump(dataset[0:370000],handle,pkl.HIGHEST_PROTOCOL)
    pkl.dump(labels[0:370000], handle, pkl.HIGHEST_PROTOCOL)
with open('data/testing_data.pickle', 'wb') as handle:
    pkl.dump(dataset[370000:],handle,pkl.HIGHEST_PROTOCOL)
    pkl.dump(labels[370000:], handle, pkl.HIGHEST_PROTOCOL)