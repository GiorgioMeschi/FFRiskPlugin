
from qgis.core import (QgsProcessing,
                       QgsGeometry, QgsVectorLayer
                       )


import processing
import numpy as np
from osgeo import gdal 
from .helpers import ProcessingHelper
import os

class PreprocessingRiskInputs:
    
    def __init__(self, context, feedback, crs):
        
        self.context = context
        self.feedback = feedback
        self.buffer_interface = 3
        self.crs = crs
        

        
     
    def eval_layer_centroid(self, layer):
        
        layer.startEditing()
        
        # Iterate over the features in the layer
        for f in layer.getFeatures():
            
            # Get the centroid of the feature's geometry
            pt = f.geometry().centroid().asPoint()
            # Create a new point geometry
            geometry = QgsGeometry.fromPointXY(pt)
            # Set the feature's geometry to the new point geometry
            f.setGeometry(geometry)
            # Update the layer
            layer.updateFeature(f)
        
        # Save the changes
        layer.commitChanges()
        
        return layer
    

    def process_single_poi(self, poi_layer, dem, feature_name_list, fieldname = 'fclass'):
        
        helper = ProcessingHelper(self.context, self.feedback)
        
        # make a selection of features I need
        for f in poi_layer.getFeatures():
            if f[fieldname] in feature_name_list:
                poi_layer.select(f.id())
        
        selected_features = poi_layer.selectedFeatures()
        
        # Create a new memory layer to store the selected features
        memory_layer = QgsVectorLayer("Polygon", "selected_points", "memory")
        
        # Add the selected features to the memory layer
        memory_layer.dataProvider().addFeatures(selected_features)     
        # centroids
        memory_layer = PreprocessingRiskInputs(self.context, self.feedback, self.crs).eval_layer_centroid(memory_layer)
                        
        poi_layer.removeSelection()
        memory_layer.removeSelection() 
        
        return memory_layer
    
    def process_single_linear(self, poi_layer, dem, feature_names, fieldname = 'fclass'):
        
        helper = ProcessingHelper(self.context, self.feedback)
        
        # make a selection of features I need
        for f in poi_layer.getFeatures():
            if f[fieldname] in feature_names:
                poi_layer.select(f.id())
        
        selected_features = poi_layer.selectedFeatures()

        # Create a new memory layer to store the selected features
        memory_layer = QgsVectorLayer("LineString", "selected_points", "memory")
        
        # Add the selected features to the memory layer
        memory_layer.dataProvider().addFeatures(selected_features)        
                
        poi_layer.removeSelection()
        memory_layer.removeSelection() 
        
        return memory_layer
    
    
    
    def preprocessing_poi(self, poi_layer, exposed_table, name_field, crs, dem, out_file_list):
        
        # helper = ProcessingHelper(self.context, self.feedback)
        preprocess_exposure = PreprocessingRiskInputs(self.context, self.feedback, self.crs)
        helper = ProcessingHelper(self.context, self.feedback)
        
        try:
            poi_layer.removeSelection()
        except:
            pass
        
        cols = 2
        nrows =  int(len(exposed_table)/cols)
        poi_table = np.array(exposed_table).reshape(nrows, cols)
        
        names_poi = poi_table[:,0]
        features_poi = poi_table[:,1]
        
        
        # processing of POI
        list_shp_paths = list()
        list_poi_names = list()
        for poiname, feature_name_list, out_path in zip(names_poi, features_poi, out_file_list):
            
            print(f'i am {poiname}')
            feature_name_list = feature_name_list.split(',')
            print(feature_name_list)

            single_poi_layer = preprocess_exposure.process_single_poi(poi_layer, dem, feature_name_list, fieldname = name_field)
            # duplicate the layer with all the info in order to ba saved correctly, it maintains its crs
            materialized_shape = helper.duplicate_point_layer(single_poi_layer, crs = poi_layer.crs().authid())
            # now reproject the layer
            poi_layer_res = helper.reproject_vector_layer(materialized_shape, prj = crs, to_path = out_path)
            
            shp_path = poi_layer_res['OUTPUT']
            
            list_shp_paths.append(shp_path)
            list_poi_names.append(poiname)
        
        return list_shp_paths, list_poi_names

        
    def preprocessing_linear(self, layer, exposed_table, name_field, crs, dem, out_file_list):
        
        # helper = ProcessingHelper(self.context, self.feedback)
        preprocess_exposure = PreprocessingRiskInputs(self.context, self.feedback, self.crs)
        helper = ProcessingHelper(self.context, self.feedback)
        
        try:
            layer.removeSelection()
        except:
            pass
        
        cols = 2
        nrows =  int(len(exposed_table)/cols)
        table = np.array(exposed_table).reshape(nrows, cols)
        
        names = table[:,0]
        features = table[:,1]
        
        
        # processing of rodas
        list_shp_paths = list()
        list_roads_names = list()
        for rname, feature_name_list, out_path in zip(names, features, out_file_list):
            
            print(f'i am {rname}')
            feature_name_list = feature_name_list.split(',')
            print(feature_name_list)

            single_poi_layer = preprocess_exposure.process_single_linear(layer, dem, feature_name_list, fieldname = name_field)
            # duplicate the layer with all the info in order to ba saved correctly, it maintains its crs
            materialized_shape = helper.duplicate_linear_layer(single_poi_layer, crs = layer.crs().authid())
            print('done')
            # now reproject the layer
            poi_layer_res = helper.reproject_vector_layer(materialized_shape, prj = crs, to_path = out_path)
            
            shp_path = poi_layer_res['OUTPUT']
                        
            list_shp_paths.append(shp_path)
            list_roads_names.append(rname)
        
        return list_shp_paths, list_roads_names
        
      
    def preprocessing_vegetation(self, fuel_model_arr, reference_ras,  out_dirpath):
        
        helper = ProcessingHelper(self.context, self.feedback)
        
        fuel_model_arr = fuel_model_arr.astype(str)
        # fuel_arrs = list()
        for code in np.unique(fuel_model_arr):
            print(f'I am code {code}')
            fuel_exposed = np.where(fuel_model_arr == code, 1, 0)
            out_path = os.path.join(out_dirpath, f'fuel_model_{code}')
            print('save fuel code')
            print(out_path)
            helper.save_temporary_array(fuel_exposed, reference_ras, out_path)
            
      
        
    def evaluate_urban_interface(self, veg_arr, urb_codes):
        
        helper = ProcessingHelper(self.context, self.feedback)
        
        # making sure veg codes and array of veg classes are integer
        veg_arr = veg_arr.astype(int)
        
        
        # print('uniques veg: ', np.unique(veg_arr))
        urb_mask = np.where( np.isin(veg_arr, urb_codes) , 1, 0)    # np.char.startswith(veg_arr_str, '1')
        print('uniques urbans: ', np.unique(urb_mask, return_counts = True))
        
        # apply a buffer to urban areas so that interface will be taken into account
        urb_interface_buff = helper.raster_buffer(urb_mask, buffer_window = self.buffer_interface)
        print('uniques urbans after buffer: ', np.unique(urb_interface_buff, return_counts = True))
        
        interface = np.where(urb_mask == 1, 0, urb_interface_buff)

        return interface
      
        
      
            

 #%%
        
        
        
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
