# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:12:12 2024
@author: dhe73118
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import math
'''Filelocations and Parameters'''
#All the csv files will be read in the folder given below and the results will be saved in the same folder
directory_path=r"C:\Users\DHENS\Desktop\Test Lifetime Fitting\IOF553_fitting test\lifetimetest"
threshold_value=6000
lifetime_guess_1=100e-5
lifetime_guess_2=400e-5

#Required plot region
if not os.path.exists(os.path.join(directory_path,'Lifetime_Graphs')):
    # Create the folder
    os.makedirs(os.path.join(directory_path, 'Lifetime_Graphs'))
else:
    print("Lifetime_Graphs already exists")

class Analysis:
   
   
    def tri_contour(self, x_values, y_values, lifetime_values, intensity_values, directory_path):
        #intensity_values=linearise(intensity_values)
        file_path=os.path.join(directory_path,'Lifetime_Graphs', 'Lifetime_map.png')
       
        #To convert to numpy array
        intensity_values=np.array(intensity_values)
        #Normalise the intensity values to the range 0 to 1
        norm_intensity_values=(intensity_values - np.min(intensity_values)) / (np.max(intensity_values) - np.min(intensity_values))
        # Create a Delaunay triangulation
        if len(norm_intensity_values) != len(lifetime_values):
            raise ValueError("The length of intensity_values must match the length of lifetime_values.")
        triangle = tri.Triangulation(x_values, y_values)
   
        # Create a tricontourf plot
        plt.tricontourf(triangle, lifetime_values, levels=20, cmap='viridis', alpha=norm_intensity_values)
   
     
        plt.colorbar(label='Lifetime')
        plt.xlabel('X Position, um')
        plt.ylabel('Y Position, um')
        plt.title('Lifetime map across surface')
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.show()

 
    def find_trigger_point(self, intensity, raster=25):
        N = len(intensity)
        startindex = 0
        if N > raster:
            intensity -= np.mean(intensity[-int(N/raster):-100])  # offset
            m = np.mean(intensity[0:int(N/raster)])
            s = math.sqrt(np.var(intensity[0:int(N/raster)]))
            startindex = int(N / raster)
            for i in range(int(N / raster), int(N - N / raster), int(N / raster)):
                nm = np.mean(intensity[i:int(i + N / raster)])
                if nm < m * 0.95:
                    startindex = int(i + N / raster / 2)
                    break
        return startindex


    def exp_decay_model(self, x,x_0, A1, A2, t1,t2, y0):
        return A1 * np.exp(-((x-x_0)/ t1))+ A2 * np.exp(-((x-x_0) / t2)) + y0
   
    def fitting_curve(self, data, directory_path, x_values, y_values):
            file_name = f'x{x_values}_y{y_values}_Lifetime'
            filepath = os.path.join(directory_path, 'Lifetime_Graphs', file_name)

            lifetime_array = data.index.values
            intensity_values = data['intensity'].values
            N = int(intensity_values.shape[0])
            offset_value=np.mean(intensity_values[-int(N/10):-2])
            startindex =1020 #
            #startindex=self.find_trigger_point(intensity_values)
           
            # Split the data: trigger points (before startindex) and decay curve (after startindex)
            lifetime_array_trigger = lifetime_array[:startindex]
            intensity_values_trigger = intensity_values[:startindex]
           
            # Points after the trigger for fitting
            lifetime_array_trimmed = lifetime_array[startindex:]
            intensity_values_trimmed = intensity_values[startindex:]

            x_0 = lifetime_array_trimmed[0]
            print('x0',x_0)
            amplitude_guess_1 = intensity_values[0]
            amplitude_guess_2 = intensity_values_trimmed[0]
            #amplitude_array = np.full(len(lifetime_array_trimmed), max_intensity_value)

            initial_guesses = [x_0, amplitude_guess_1, amplitude_guess_2, lifetime_guess_1, lifetime_guess_2, offset_value]
           
            if np.max(intensity_values_trimmed) - np.min(intensity_values_trimmed) > threshold_value:  # Threshold value
                try:
                    params, covariance = curve_fit(self.exp_decay_model, lifetime_array_trimmed, intensity_values_trimmed, p0=initial_guesses, bounds=((0, 0, 0, 0, 0, -np.inf), (np.inf, max(intensity_values), max(intensity_values), np.inf, np.inf, np.inf)))
                except RuntimeError:
                    params = [0, 0, 0, 0, 0, 0]
            else:
                params = [0, 0, 0, 0, 0, 0]

            # Extract fitting parameters
            x_offset, A1, A2, t1, t2, y0 = params
            y_fit = self.exp_decay_model(lifetime_array_trimmed, x_offset, A1, A2, t1, t2, y0)

            # Plot original data
            plt.figure(figsize=(8, 6))
           
            # Plot all the data points including trigger points
            plt.plot(lifetime_array, np.log(intensity_values), color='blue', label='Data points')

            # Highlight trigger points in a different color
            plt.scatter(lifetime_array_trigger, np.log(intensity_values_trigger), color='orange', label='masked points')
            t1_microseconds = round(t1 * 1e6, 1)
            t2_microseconds = round(t2 * 1e6, 1)

           
            plt.plot(lifetime_array_trimmed, np.log(y_fit), 'r-', label=f'Fit, T_1={t1_microseconds} μs, T_2={t2_microseconds} μs')
            plt.xlabel('Time (ms)')
            plt.ylabel('Intensity(log)')
            plt.title('Lifetime fitting with trigger points')
            plt.legend()
            plt.grid(True)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            #plt.show()

            print(f"Lifetime, t1: {t1}")
            print(f"Lifetime, t2: {t2}")
            print("A1:", A1)
            print("A2:", A2)
            if t2 == 0:
                t_cm = 0
            else:
                t_cm = (A1 * t1**2 + A2 * t2**2) / (A1 * t1 + A2 * t2)

            print("Average:", t_cm)
            return t1, t2, amplitude_guess_1, t_cm
   
    def reading_csvfiles_ranges(self, directory_path):
        combined_data = pd.DataFrame()
        column_name = []
       
        # Iterate over each CSV file in the directory
        i = 1
        for file_name in os.listdir(directory_path):
            if file_name.endswith("_OZI.CSV"):
                # Read the CSV file into a DataFrame
                file_path = os.path.join(directory_path, file_name)
                df = pd.read_csv(file_path, skiprows=1, sep=',', header=None, index_col=False)
                column_name.append(file_name)
                if i == 1:
                    # Append the data to the combined DataFrame
                    combined_data = pd.concat([combined_data, df], axis=1)
                    i = 0
                else:
                    # Append only the intensity column to the combined DataFrame
                    combined_data = pd.concat([combined_data, df.iloc[:, 1]], axis=1)
       
        combined_data = combined_data.rename(columns={combined_data.columns[0]: 'time(s)'})
        for i in range(1, len(column_name) + 1):
            combined_data.columns.values[i] = column_name[i - 1]
       
       
        # Filter the DataFrame to include rows within the specified  range
       
        # Melt the DataFrame to long format
        melted_df = pd.melt(combined_data, id_vars=['time(s)'], value_vars=column_name, var_name='position', value_name='intensity')
        melted_df[['x', 'y']] = melted_df['position'].str.split('_y', expand=True)
        melted_df['x'] = melted_df['x'].str[1:].astype(int)
        melted_df['y'] = melted_df['y'].str.split('_OZI', expand=True)[0].astype(int)
        melted_df.set_index('time(s)', inplace=True)
        melted_df.drop('position', axis=1, inplace=True)

       
        x_values = []
        y_values = []

        # Iterate over each file in the folder to get the positions x and y
        for filename in os.listdir(directory_path):
            # Check if the file name is a CSV file
            if filename.lower().endswith(".csv"):
                # Extract x and y values from the file name
                x_value, y_value = filename.split("_")[0][1:], filename.split("_")[1][1:]
                # Append x and y values to their respective lists
                x_values.append(int(x_value))
                y_values.append(int(y_value))

       
       # Assuming you have already defined unique_x_values and unique_y_values
        lifetime_values_t1=[]
        lifetime_values_t2=[]
        intensity_values=[]
        t_cm_values=[]
       
        for i in range(len(x_values)):

            # Filter the DataFrame for the specific x and y values
            filtered_df_ = melted_df[(melted_df['x'] == x_values[i]) & (melted_df['y'] == y_values[i])]
            # Call the function to fit with lmfit (replace with your actual function)
            lifetime_t1, lifetime_t2, intensity_val, t_centre_of_mass=self.fitting_curve(filtered_df_, directory_path, x_values[i], y_values[i])
            lifetime_values_t1.append(lifetime_t1)
            lifetime_values_t2.append(lifetime_t2)
            intensity_values.append(intensity_val)
            t_cm_values.append(t_centre_of_mass)
           
   
        # Create a DataFrame with the lists as columns
        df = pd.DataFrame()
        df['x'] =  x_values
        df['y'] = y_values
        df['I0'] = intensity_values
        df['tau_1'] = lifetime_values_t1
        df['tau'] = lifetime_values_t2
        df['t_cm']=t_cm_values
        # Save to CSV file
        df.to_csv(os.path.join(directory_path,'Lifetime_Graphs', 'auswertung.csv'), index=False)
        print("Data saved to auswertung.csv")
        #self.tri_contour(x_values, y_values, lifetime_values,intensity_values, directory_path)

ana=Analysis()
#ana.reading_csvfiles(directory_path=r"C:\Users\DHENS\Desktop\Python Analysis\20240506_143225_My Code", wavelength_selected=800)
ana.reading_csvfiles_ranges(directory_path)