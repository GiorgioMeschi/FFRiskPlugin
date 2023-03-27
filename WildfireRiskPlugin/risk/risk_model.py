
from qgis.core import QgsField                       
from PyQt5.QtCore import QVariant

import numpy as np
from osgeo import gdal 
from WildfireRiskPlugin.helpers import ProcessingHelper

import struct

class RiskEval():
    
    def __init__(self, context, feedback):
        self.context = context
        self.feedback = feedback
        self.helper = ProcessingHelper(self.context, self.feedback)
        self.buffer_window = 3
        self.upper_limit_perc = 0.6
        self.damages_total = [0, 0.25, 0.5, 0.75, 1] # classes on normalized array
        self.damages_exposure_specific = [0, 25, 50, 75, 100] # degree of damage of POI (monetary or importance)
    
    

                                
    def evaluate_potential_damage(self, intens_arr, array, V, E):
        '''
        array is the binary array of exposed element
        V is list of vulnerabilities for a single exposed eleemnt (a value for each H value)
        E is a value of exposed element 
        '''
        
        
        intens_clipped = np.where(array == 1, intens_arr, 0)
        # remove invalid codes 
        
        
        # assign vulnerability to each intensity level 
        H_levels = [1, 2, 3, 4, 5, 6]
        V_asset_grid = self.helper.reclassify_raster_searchsort(intens_clipped, V, H_levels, nodata = 0) # no data of I map, return no data: -1
        
        # assign degree of damage multiplying V and importance value
        damage_degree_map = V_asset_grid * E
        damage_degree_map = np.where(damage_degree_map < 0, -1, damage_degree_map) # 0 damage exists, -1 is nodata
        
        return damage_degree_map
    
    def assign_damage_to_shapefile(self, layer, array, reference_raster, col_name = 'Damage'):
        
        helper = ProcessingHelper(self.context, self.feedback)
        

        raster_path = helper.save_temporary_array(array, reference_raster)
        # open the raster dataset
        dataset = gdal.Open(raster_path)
        # print(dataset)
        # get the raster band
        band = dataset.GetRasterBand(1)
        # print(band)
        
        # save info for addressing pixel indeces from coordinates
        gt_forward = dataset.GetGeoTransform()
        gt_reverse = gdal.InvGeoTransform(gt_forward) 
        
        # add the value to the shapefile
        layer_provider = layer.dataProvider()
        # column = [field.name() for field in layer.fields()][0]
        
        layer_provider.addAttributes([QgsField(col_name, QVariant.Double)])
        layer.updateFields()
        
        layer.startEditing()
        for feature in layer.getFeatures():
            
            geometry = feature.geometry()
            # get the centroid of the polygon
            centroid = geometry.centroid().asPoint()
            x = int(centroid.x())
            y = int(centroid.y())

            # read the raster value at the given coordinates 
            px, py = gdal.ApplyGeoTransform(gt_reverse, x, y)
            structval = dataset.ReadRaster(int(px), int(py), 1, 1, buf_type=gdal.GDT_UInt16)
            
            # use struct library to retreive info of raster value given couple of indeces
            try:
                val = struct.unpack('h' , structval)[0]
            except TypeError:
                print('found type error')
                val = 0
            
            # print(f'pixel value of asset: {val}')

            col = feature.id()
            index = len([field.name() for field in layer.fields()])-1
            attr_val = {index:val}
            layer_provider.changeAttributeValues({col:attr_val})
        layer.commitChanges()
        
        # close the dataset
        dataset = None
        
        return layer
    
    def assign_name_to_shapefile(self, layer, col_name, feature_name):
        
        
        # add the value to the shapefile
        layer_provider = layer.dataProvider()
        # column = [field.name() for field in layer.fields()][0]
        
        layer_provider.addAttributes([QgsField(col_name, QVariant.String)])
        layer.updateFields()
        
        layer.startEditing()
        for feature in layer.getFeatures():
            
            val = feature_name
            
            col = feature.id()
            index = len([field.name() for field in layer.fields()])-1
            attr_val = {index:val}
            layer_provider.changeAttributeValues({col:attr_val})
        layer.commitChanges()
        
        
        return layer
        
    def classify_total_damage(self, potential_degree_of_damage):
        
        potential_degree_of_damage = np.where(potential_degree_of_damage < 0, -1, potential_degree_of_damage)
        
        # normalization with % of max value
        potential_degree_of_damage = potential_degree_of_damage / (np.max(potential_degree_of_damage) * self.upper_limit_perc) 
        
        potential_degree_of_damage = np.clip(potential_degree_of_damage, 
                                             a_min = potential_degree_of_damage.min(), 
                                             a_max = 1)
        
        # classify in 5 classes of potential damage
        bounds = self.damages_total
        PDD_CL = self.helper.raster_classes_vals(potential_degree_of_damage, bounds, nodata = -1) # nodata 0 will be coverted in class: 0 in output
        
        return PDD_CL

        
    def process_hazards(self, hazard_arr):
        '''
        average hazard over a sliding windows
        '''
        helper = ProcessingHelper(self.context, self.feedback)
        
        # get class 1 when hazard is no data --> this is just for doing the sliding moving average
        hazard_arr_modified = np.where(hazard_arr == 0, 1, hazard_arr)
        
        haz_arr_filter = helper.average_sliding_windows(hazard_arr_modified, self.buffer_window)
        haz_arr_filter = np.round(haz_arr_filter).astype(int)
        
        # put no data (class 0) if area were not burnable
        # haz_arr_filter = np.where(hazard_arr == 0, 0, haz_arr_filter)
        
        print(f'uniques of processed array: {np.unique(haz_arr_filter)}')
        
        return haz_arr_filter
        
    # def risk_matrix(self, arr1, arr2):  #arr1 takes values on the rows, arr2 on the columns
    #     '''
    #     evaluate the the risk with contingency matrix between haz and damages
        
    
    #     # H / D	1	2	3	4
    #     #   1	1	1	2	2
    #     #   2	1	1	2	3
    #     #   3	1	2	2	3
    #     #   4	1	3	3	4
    #     #   5	2	3	4	4
    #     #   6	2	4	4	4
        
    #     '''
    	
    
    #     conditions  = [((arr1 == 1) & (arr2 == 1)) +
    #                    ((arr1 == 1) & (arr2 == 2)) +
    #                    ((arr1 == 2) & (arr2 == 1)) +
    #                    ((arr1 == 4) & (arr2 == 1)) +
    #                    ((arr1 == 2) & (arr2 == 2)) +
    #                    ((arr1 == 3) & (arr2 == 1)),
                      
    #                    ((arr1 == 1) & (arr2 == 3)) +
    #                    ((arr1 == 1) & (arr2 == 4)) +
    #                    ((arr1 == 2) & (arr2 == 2)) +
    #                    ((arr1 == 2) & (arr2 == 3)) +
    #                    ((arr1 == 3) & (arr2 == 2)) +
    #                    ((arr1 == 3) & (arr2 == 3)) +
    #                    ((arr1 == 5) & (arr2 == 1)) +
    #                    ((arr1 == 6) & (arr2 == 1)),
                       
    #                    ((arr1 == 2) & (arr2 == 4)) +
    #                    ((arr1 == 3) & (arr2 == 4)) +
    #                    ((arr1 == 4) & (arr2 == 2)) +
    #                    ((arr1 == 4) & (arr2 == 3)) +
    #                    ((arr1 == 5) & (arr2 == 2)),
                       
    #                    ((arr1 == 4) & (arr2 == 4)) +
    #                    ((arr1 == 5) & (arr2 == 3)) +
    #                    ((arr1 == 5) & (arr2 == 4)) +
    #                    ((arr1 == 6) & (arr2 == 2)) +
    #                    ((arr1 == 6) & (arr2 == 3)) +
    #                    ((arr1 == 6) & (arr2 == 4)), 
                       
    #                    ]
        
    #     classes = [ 1, 2, 3, 4 ] # low medium high critical 
    #     out_arr = np.select(conditions, classes, default=0)
          
        
    #     return out_arr
    
    
    def risk_matrix(self, arr1, arr2):  #arr1 takes values on the rows, arr2 on the columns
        '''
        evaluate the the risk with contingency matrix between haz and damages
        
    
        # H / D	1	2	3	4
        #   1	1	1	1	1
        #   2	1	1	2	2
        #   3	1	2	3	3
        #   4	2	3	3	4
        #   5	3	4	4	4
        #   6	4	4	4	4
        
        '''
    	
    
        conditions  = [((arr1 == 1) & (arr2 == 1)) +
                       ((arr1 == 1) & (arr2 == 2)) +
                       ((arr1 == 1) & (arr2 == 3)) +
                       ((arr1 == 1) & (arr2 == 4)) +
                       ((arr1 == 2) & (arr2 == 1)) +
                       ((arr1 == 2) & (arr2 == 2)) +
                       ((arr1 == 3) & (arr2 == 1)),
                       
                       ((arr1 == 2) & (arr2 == 3)) +
                       ((arr1 == 2) & (arr2 == 4)) +
                       ((arr1 == 3) & (arr2 == 2)) +
                       ((arr1 == 4) & (arr2 == 1)),
                       
                       ((arr1 == 3) & (arr2 == 3)) +
                       ((arr1 == 3) & (arr2 == 4)) +
                       ((arr1 == 4) & (arr2 == 2)) +
                       ((arr1 == 4) & (arr2 == 3)) +
                       ((arr1 == 5) & (arr2 == 1)),
                       
                       ((arr1 == 4) & (arr2 == 4)) +
                       ((arr1 == 5) & (arr2 == 2)) +
                       ((arr1 == 5) & (arr2 == 3)) +
                       ((arr1 == 5) & (arr2 == 4)) +
                       ((arr1 == 6) & (arr2 == 1)) +
                       ((arr1 == 6) & (arr2 == 2)) +
                       ((arr1 == 6) & (arr2 == 3)) +
                       ((arr1 == 6) & (arr2 == 4)),

                       ]
        
        classes = [ 1, 2, 3, 4 ] # low medium high critical 
        out_arr = np.select(conditions, classes, default=0)
          
        
        return out_arr
    
                                    
    def element_specific_risk(self, damage_degree_map):
        
        # classify the map of specific damage with custom classes of damage
        damage_degree_map = np.where(damage_degree_map < 0, -1, damage_degree_map)
        
        # classify in 5 classes of potential damage
        element_risk_arr = self.helper.raster_classes_vals(damage_degree_map, self.damages_exposure_specific, nodata = -1) # nodata -1 will be coverted in class: 0 in output
        
        print(f'uniques specific risk classes: {np.unique(element_risk_arr)}')
        
        return element_risk_arr 
                        
                                                                             
                                    
                                    
                                    
                                    
                                    
                                    
                   
       
        #%%
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
