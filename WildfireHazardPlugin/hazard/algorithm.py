# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:17:56 2022

@author: Giorg
"""

from qgis.core import QgsProcessing
import processing

import numpy as np
from osgeo import gdal 
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score

from WildfireHazardPlugin.helpers import ProcessingHelper



class RFForestAlgorithm:
    
    def __init__(self, context, feedback):
        self.context = context
        self.feedback = feedback
        self.number_of_trees = 100
        
    def preprocessing(self, dem_arr, dem_nodata, fires_arr, veg_arr, 
                      nb_codes_list, slope_arr, northing_arr, easting_arr, 
                      other_layers_dict):
        
        # mask the vegetation
        veg_arr = veg_arr.astype(int)
        veg_arr_str = veg_arr.astype(str)        
        for i in nb_codes_list:
            veg_arr_str = np.where(veg_arr_str == i, '0', veg_arr_str)
        veg_mask = np.where(veg_arr_str == '0', 0, 1)
                
        # now a complete the mask adding dem existence
        mask = (veg_mask == 1) & (dem_arr != dem_nodata)
        
        # evaluation of perc just in vegetated area, non vegetated are grouped in code 0
        veg_f = veg_arr_str.astype(np.float)
        veg_int = veg_f.astype(int)
        veg_int = np.where(veg_mask == 1, veg_int, 0)
        window_size = 2
        types = np.unique(veg_int)
        types_presence = {}
        
        counter = np.ones((window_size*2+1, window_size*2+1))
        take_center = 1
        counter[window_size, window_size] = take_center 
        counter = counter / np.sum(counter)

        # perc --> neightboring vegetation generation                
        for t in types:
            density_entry = 'perc_' + str(int(t)) 
            print(f'Processing vegetation density {density_entry}')
            types_presence[density_entry] = 100 * signal.convolve2d(veg_int==t, counter, boundary='fill', mode='same')
        
        # dict of layers for ML dataset generation            
        data_dict = {
            'dem': dem_arr,
            'slope': slope_arr,
            'north': northing_arr,
            'east': easting_arr,
            'veg': veg_arr_str,
        }

        for layer_name, layer_arr in other_layers_dict.items():
            data_dict[layer_name] = layer_arr
    
        data_dict.update(types_presence)

        # creaate X and Y datasets
        n_pixels = len(dem_arr[mask])
        n_features = len((data_dict.keys()))
        X_all = np.zeros((n_pixels, n_features))
        Y_all = fires_arr[mask]

        self.feedback.pushInfo('Creating dataset for RandomForestClassifier')
        columns = data_dict.keys()
        for col, k in enumerate(data_dict):
            print(f'Processing column: {k}')
            data = data_dict[k]
            X_all[:, col] = data[mask]

        return X_all, Y_all, dem_arr, mask, columns 
    
    def train(self, X_all, Y_all, percentage): 
        
        # filter df taking info in the burned points
        fires_rows = Y_all != 0
        X_presence = X_all[fires_rows]
        
        # sampling training set       
        self.feedback.pushInfo(' I am random sampling the dataset ')
        # reduction of burned points --> reduction of training points       
        reduction = int((X_presence.shape[0]*percentage)/100)
        self.feedback.pushInfo(f"reducted df points: {reduction} of {X_presence.shape[0]}")
        
        # sampling and update presences 
        X_presence_indexes = np.random.choice(X_presence.shape[0], size=reduction, replace=False)
        X_presence = X_presence[X_presence_indexes, :]         
        # select not burned points
        X_absence = X_all[~fires_rows]
        X_absence_choices_indexes = np.random.choice(X_absence.shape[0], size=X_presence.shape[0], replace=False)
        X_pseudo_absence = X_absence[X_absence_choices_indexes, :]
        # create X and Y with same number of burned and not burned points
        X = np.concatenate([X_presence, X_pseudo_absence], axis=0)
        Y = np.concatenate([np.ones((X_presence.shape[0],)), np.zeros((X_presence.shape[0],))])
        # create training and testing df with random sampling
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        
        self.feedback.pushInfo(f'Running RF on data sample: {X_train.shape} ')
        model  = RandomForestClassifier(n_estimators=self.number_of_trees, verbose = 2)
        
        return model, X_train, X_test, y_train, y_test


    def print_stats(self, model, X_train, y_train, X_test, y_test, columns):
        
        # fit model 
        model.fit(X_train, y_train)
        # stats on training df
        p_train = model.predict_proba(X_train)[:,1]
        auc_train = roc_auc_score(y_train, p_train)
        self.feedback.pushInfo(f'AUC score on train: {auc_train:.2f}')
        
        # stats on test df
        p_test = model.predict_proba(X_test)[:,1]
        auc_test = roc_auc_score(y_test, p_test)
        self.feedback.pushInfo(f'AUC score on test: {auc_test:.2f}')
        mse = mean_squared_error(y_test, p_test)
        self.feedback.pushInfo(f'MSE: {mse:.2f}')
        p_test_binary = model.predict(X_test)
        accuracy = accuracy_score(y_test, p_test_binary)
        self.feedback.pushInfo(f'accuracy: {accuracy:.2f}')
        
        # features impotance
        self.feedback.pushInfo('I am evaluating features importance')       
        imp = model.feature_importances_
        
        perc_imp_list = list()
        list_imp_noPerc = list()
        
        # separate the perc featuers with the others 
        for i,j in zip(columns, imp):
            if i.startswith('perc_'):
                perc_imp_list.append(j)
            else:
                list_imp_noPerc.append(j)
                
        # aggregate perc importances
        perc_imp = sum(perc_imp_list)
        # add the aggregated result
        list_imp_noPerc.append(perc_imp)
        
        # list of columns of interest
        cols = [col for col in columns if not col.startswith('perc_')]
        cols.append('perc')
        
        # print results
        self.feedback.pushInfo('importances')
        dict_imp = dict(zip(cols, list_imp_noPerc))
        dict_imp_sorted = {k: v for k, v in sorted(dict_imp.items(), 
                                                   key=lambda item: item[1], 
                                                   reverse=True)}
        for i in dict_imp_sorted:
            self.feedback.pushInfo('{} : {}'.format(i, round(dict_imp_sorted[i], 2)))

                
    
    def get_results(self, model, X_all, dem_arr, dem_raster, mask, susc_path):
        
        helper = ProcessingHelper(self.context, self.feedback)
        
        # prediction over all the points
        Y_out = model.predict_proba(X_all)
        # array of predictions over the valid pixels 
        Y_raster = np.zeros_like(dem_arr)
        Y_raster[mask] = Y_out[:,1]
        
        # clip susc where dem exsits
        Y_raster[~mask] = -1
                
        # save as raster file
        helper.saverasternd(dem_raster, susc_path, Y_raster)
        
        return Y_raster, susc_path

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
