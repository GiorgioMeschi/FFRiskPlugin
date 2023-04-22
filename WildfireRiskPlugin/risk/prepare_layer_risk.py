
from qgis.core import (QgsProcessing,
                       QgsGeometry, QgsVectorLayer, QgsRasterLayer,
                       )


import processing
import numpy as np
from osgeo import gdal 
from WildfireRiskPlugin.helpers import ProcessingHelper
import os

class PrepareRiskInputs:
    
    def __init__(self, context, feedback, crs):
        
        self.context = context
        self.feedback = feedback
        self.crs = crs
        self.pop_cl_square_km = [10, 500, 1500, 3000, 1000000]


        
     
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
    

    def prepare_single_poi(self, poi_layer, dem):
        
        helper = ProcessingHelper(self.context, self.feedback)
             
        # centroids
        centroid_layer = PrepareRiskInputs(self.context, self.feedback, self.crs).eval_layer_centroid(poi_layer)
        
        # rasterize and return the array the centroid of exposed eleemnt       
        layer_result = helper.rasterize_numerical_feature(centroid_layer, dem, column=None, burn=0.0)
        layer_raster_path = layer_result['OUTPUT']
        layer_raster = gdal.Open(layer_raster_path)
        layer_arr = layer_raster.GetRasterBand(1).ReadAsArray()
        print(f'uniques raster exposed element: {np.unique(layer_arr, return_counts=True)}')
                
        
        # retunr alsoe the filtered layer: it will be used later for evalauting the specific risk
        return layer_arr, centroid_layer
    
    
    def read_exposed_table(self, exposed_table, dirpath, cols, index_filename):
        nrows =  int(len(exposed_table)/cols)
        
        print(f'ROWS: {nrows}')
       
        poi_table = np.array(exposed_table).reshape(nrows, cols)

        
        
        return poi_table, nrows
    
    def create_lists_from_table(self, table, nrows, 
                                dirpath, index_filename,
                                indexes_vulnerabilities, index_exposure, geotiff = False):
        '''
        '''
        
        list_layers_path = [os.path.join(dirpath, table[i, index_filename]) for i in range(nrows)] 
        print(list_layers_path)
        
        if geotiff:
            list_layers = [QgsRasterLayer(path, 'Layer name', 'ogr') for path in list_layers_path]
        else:
            list_layers = [QgsVectorLayer(path, 'Layer name', 'ogr') for path in list_layers_path]
        list_V = [[round(float(table[i,j]), 2) for j in indexes_vulnerabilities ] for i in range(nrows)] 
        list_E = [round(float(table[i, index_exposure]), 2) for i in range(nrows)]
        
        return list_layers_path, list_layers, list_V, list_E
    
    
    def prepare_single_linear_poly(self, layer, dem):
        
        helper = ProcessingHelper(self.context, self.feedback)
        
        print('rasterizing linear element...')
        # rasterize and return the array the centroid of exposed eleemnt       
        layer_result = helper.rasterize_numerical_feature(layer, dem, column=None, burn=0.0)
        layer_raster_path = layer_result['OUTPUT']
        layer_raster = gdal.Open(layer_raster_path)
        layer_arr = layer_raster.GetRasterBand(1).ReadAsArray()
        print(f'uniques raster linear element: {np.unique(layer_arr, return_counts=True)}')
                
        return layer_arr
    
    
    
    

    def prepare_population(self, pop_layer, ref_layer, conversion = True):
        '''
        take population array and classify it in order to procuce the risk
        population in input has to be in square km otherwise conversion wont work!
        if you dont want to apply this conversion set the parameter to false
        '''
        
        helper = ProcessingHelper(self.context, self.feedback)
        # process = PreprocessingRiskInputs(self.context, self.feedback, self.crs, self.metadata_csv_file)

        # pop reprojection
        pop_repr = helper.reproject_layer(pop_layer, ref_layer)
        pop_veg = pop_repr['OUTPUT']
       
        pop_raster = gdal.Open(pop_veg)
        # read file as array
        pop_arr = pop_raster.GetRasterBand(1).ReadAsArray()  
        
        # considering no data up to 10 person over 1 km2
        pop_arr = np.where(pop_arr < 10, 0, pop_arr)
        
        # conversion factor of pop inc ase change of resolution is needed
        p = ref_layer.dataProvider().dataSourceUri()       
        ref_raster = gdal.Open(p)
        # this has to be in meters
        res = ref_raster.GetGeoTransform()[1]
        res_km2 = res**2 / 1000000
        # factor will be divided by the number of people in a km2 pixel size
        conversion_factor = 1/res_km2
        
        # redefine classes thresolds based on the resolution of analisis
        if conversion:
            pop_arr /= conversion_factor 
            # rescale also the classes 


        # in any case I rescale the classes because they are per square km by default     
        pop_classes = [ i / conversion_factor for i in self.pop_cl_square_km] 
        
        pop_arr_cl = helper.raster_classes_vals(pop_arr,  pop_classes, nodata = 0, norm = False)
        
        
        return pop_arr_cl
        


            

 #%%
        
        
        
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
