from qgis.core import QgsProcessing, QgsField, QgsProcessingUtils, QgsVectorFileWriter, QgsRasterFileWriter, QgsVectorLayer
from PyQt5.QtCore import QVariant, QTemporaryFile
import processing

import numpy as np
from osgeo import gdal 
from scipy import ndimage
from scipy.ndimage import maximum_filter as maxf2D
from scipy.ndimage import uniform_filter

class ProcessingHelper:

    def __init__(self, context, feedback):
        self.context = context
        self.feedback = feedback
        
        
    def read_arr_after_qgis_process(self, processed_layer):
        
        path = processed_layer['OUTPUT']
        raster = gdal.Open(path)
        array = raster.GetRasterBand(1).ReadAsArray()
        
    def evalaute_probabilities(self, hazard_arr, fires_arr, len_years):

        P = np.zeros(hazard_arr.shape)
        h_cl = list(range(1,13))
        for h in h_cl:
            # percentage of fires in each hazard class
            burned_pixels = np.where(hazard_arr == h, fires_arr, 0).sum()
            all_pixels = np.where(hazard_arr == h, 1, 0).sum()
            P = np.where(hazard_arr == h, (burned_pixels / len_years) / all_pixels, P)

        return P
        
    def rasterize_numerical_feature(self, layer, reference_layer, column=None, burn=0.0):
        '''
        rasterize a shapefile
        reference layer is a parameterAsRasterLayer object
        layer is parameterAsVectorLayer object
        '''
        # createthecolumwith all ones to rasterize if no  colsare selected
        if column is None:  
            layer_provider = layer.dataProvider()
            # column = [field.name() for field in layer.fields()][0]
            
            layer_provider.addAttributes([QgsField('myCol', QVariant.Int)])
            layer.updateFields()
            column = [field.name() for field in layer.fields()][-1]
            print(column)
            
            layer.startEditing()
            for f in layer.getFeatures():
                col = f.id()
                val = 1
                index = len([field.name() for field in layer.fields()])-1
                attr_val = {index:val}
                layer_provider.changeAttributeValues({col:attr_val})
            layer.commitChanges()
                     
        raster_layer = processing.run("gdal:rasterize", 
            dict(
                INPUT=layer,
                FIELD=column,
                # BURN=burn,
                HEIGHT=reference_layer.height(),
                WIDTH=reference_layer.width(),
                EXTENT=reference_layer.extent(),
                UNITS=0, # witdh and height are expressed as pixels
                ALL_TOUCHED = True,
                EXTRA = '-at',
                OUTPUT=QgsProcessing.TEMPORARY_OUTPUT
            ),
            context=self.context,
            feedback=self.feedback,
            is_child_algorithm=True,
        )
        
        #remove the temporarycolumn used for rasterizing
        if column == 'myCol':
            try:
                layer_provider.deleteAttributes([index])
                layer.updateFields()
            except:
                print('returned empty raster')

        return raster_layer


    def clip_raster_by_mask(self, raster_layer, shp_mask, crop_ = True):
        '''
        clip a raster with a shapefile
        raster_layer is a parameterAsRasterLayer object
        shp_mask is parameterAsVectorLayer object
        '''
       
      
        raster_layer_clipped = processing.run("gdal:cliprasterbymasklayer", 
            dict(
                INPUT = raster_layer,
                MASK = shp_mask,
                SOURCE_CRS = raster_layer.crs().authid(),
                TARGET_CRS = raster_layer.crs().authid(),
                NODATA = -9999,
                # ALPHA_BAND = False,
                CROP_TO_CUTLINE = crop_,
                KEEP_RESOLUTION = True,
                # OTPTIONS = None,
                # DATA_TYPE = 0,
                OUTPUT=QgsProcessing.TEMPORARY_OUTPUT
            ),
            context=self.context,
            feedback=self.feedback,
            is_child_algorithm=True,
        )
        
        return raster_layer_clipped

    def reproject_layer(self, layer, reference_layer):
        
        crs_ref = reference_layer.crs()
        extent = reference_layer.extent()
        provider = reference_layer.dataProvider()
        yres = reference_layer.rasterUnitsPerPixelY()
        xres = reference_layer.rasterUnitsPerPixelX()
        block = provider.block(1, extent, yres, xres)
        no_data_val = block.noDataValue()
        
        param = dict(INPUT = layer,
                     SOURCE_CRS = layer.crs(),
                     TARGET_CRS = crs_ref,
                     # resampling default Nearest neighbour
                     NODATA = no_data_val,
                     TARGET_RESOLUTION = xres,
                     TARGET_EXTENT = extent,
                     TARGET_EXTENT_CRS = crs_ref,
                     OUTPUT = QgsProcessing.TEMPORARY_OUTPUT)
        
        raster_layer = processing.run('gdal:warpreproject', param, 
                                      context=self.context,
                                      feedback=self.feedback,
                                      is_child_algorithm=True)
        
        return raster_layer


    def calculate_slope(self, layer):
        self.feedback.pushInfo(f'Calculating slope')
        raster_layer = processing.run('qgis:slope', 
            dict(
                INPUT=layer,
                Z_FACTOR=1.0,
                OUTPUT=QgsProcessing.TEMPORARY_OUTPUT
            ),
            context=self.context,
            feedback=self.feedback,
            is_child_algorithm=True
        )
            
        return raster_layer

        

    def calculate_aspect(self, layer):
        self.feedback.pushInfo(f'Calculating aspect')
        raster_layer =  processing.run('qgis:aspect',
            dict(
                INPUT=layer,
                Z_FACTOR=1.0,
                OUTPUT=QgsProcessing.TEMPORARY_OUTPUT
            ),            
            context=self.context,
            feedback=self.feedback,
            is_child_algorithm=True
        )
        
        return raster_layer
        

    def write_distance_raster(self, layer, band=1, values=[0]):
        distance_result = processing.run('gdal:proximity', 
            dict(
                INPUT=layer,
                BAND=band,
                VALUES=values,
                UNITS=0, #Georeferenced coordinates
                OUTPUT=QgsProcessing.TEMPORARY_OUTPUT
            ), 
            context=self.context,
            feedback=self.feedback
        )

        return distance_result


    def veg_aggregation(self, vector_layer, veg_arr, name_veg_col = 'veg', name_intens_col = 'aggr'):
        
        codes = list()
        intensity = list()
        
        for feature in vector_layer.getFeatures():
            codes.append(feature[name_veg_col])
            intensity.append(feature[name_intens_col])
            
        codes = [int(i) for i in codes]   
        intensity = [int(i) for i in intensity]
        
        print('veg codes:\n', codes, 'intensity:\n', intensity)
        
        veg_arr_aggr = veg_arr.astype(int)
        
        breaking = False
        for i in intensity:
            if i in codes:
                self.feedback.pushInfo('the vegetation codes must be different from intensity codes')
                breaking = True
                break
            else:
                pass
            
        if not breaking:    
            for i,j in zip(codes, intensity):         
                veg_arr_aggr = np.where(veg_arr_aggr == i, j, veg_arr_aggr)
        else:
           self.feedback.pushInfo('could not procede, fix your vegetation codes') 
        
        
        return veg_arr_aggr


            

    def susc_classes(self, susc_arr, quantiles):
        
        bounds = [0] + list(quantiles) + [1]
        
        # convert the raster map into a categorical map based on quantile values
        conditions = list()
        for i in range(0, len(quantiles)+1 ):
            # first position take also ssuc = 0, the dosnt take the low limit
            if i == 0:
                conditions.append( ((susc_arr >= bounds[i]) & (susc_arr <= bounds[i+1])) )
            else:
                conditions.append(((susc_arr > bounds[i]) & (susc_arr <= bounds[i+1])))
        
        classes = [i for i in range(1, len(bounds))]
        
        out_arr = np.select(conditions, classes, default=0)
        out_arr = out_arr.astype(np.int8())
        
        return out_arr
        


    def hazard_matrix(self, arr1, arr2):  #arr1 take values on the rows, arr2 on the columns
        '''
        	 #  s\i 1	2	3	4	
        #       1   1	2	3	4
        #       2   2	3	4	5
        #       3   3	3	5	6
        '''
    
        conditions  = [((arr1 == 1) & (arr2 == 1)),
                       
                       ((arr1 == 1) & (arr2 == 2)) +
                       ((arr1 == 2) & (arr2 == 1)),
                       
                       ((arr1 == 1) & (arr2 == 3)) +
                       ((arr1 == 2) & (arr2 == 2)) + 
                       ((arr1 == 3) & (arr2 == 1)),
                       
                       ((arr1 == 1) & (arr2 == 4)) +
                       ((arr1 == 2) & (arr2 == 3)) + 
                       ((arr1 == 3) & (arr2 == 2)),
                       
                       ((arr1 == 2) & (arr2 == 4)) +
                       ((arr1 == 3) & (arr2 == 3)),
                       
                       ((arr1 == 3) & (arr2 == 4))
                       
                       ]
        
        classes    = [1, 2, 3, 4, 5, 6]
        out_arr   = np.select(conditions, classes, default=0)
      
        
        return out_arr
    
    def hazard_matrix_V2(self, arr1, arr2):
        
        conditions  = [((arr1 == 1) & (arr2 == 1)),
                        ((arr1 == 2) & (arr2 == 1)),
                        ((arr1 == 3) & (arr2 == 1)),
                        ((arr1 == 1) & (arr2 == 2)),
                        ((arr1 == 2) & (arr2 == 2)),
                        ((arr1 == 3) & (arr2 == 2)),
                        ((arr1 == 1) & (arr2 == 3)),
                        ((arr1 == 2) & (arr2 == 3)),
                        ((arr1 == 3) & (arr2 == 3)),
                        ((arr1 == 1) & (arr2 == 4)),
                        ((arr1 == 2) & (arr2 == 4)),
                        ((arr1 == 3) & (arr2 == 4)),
                                            
                        ]
    
        classes    = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        out_arr   = np.select(conditions, classes, default=0)
    
        return out_arr            

    def saverasternd(self, gdal_data, filename, raster):
        rows = gdal_data.RasterYSize
        cols = gdal_data.RasterXSize

        outDs = gdal.GetDriverByName("GTiff").Create(filename, cols, rows, int(1), gdal.GDT_Float32) 
        outBand = outDs.GetRasterBand(1)

        # write the data
        outBand.WriteArray(raster, 0, 0)
        # flush data to disk, set the NoData value and calculate stats
        outBand.FlushCache()
        # outBand.SetNoDataValue(-9999)

        # georeference the image and set the projection
        outDs.SetGeoTransform(gdal_data.GetGeoTransform())
        outDs.SetProjection(gdal_data.GetProjection())

    def save_temporary_array(self, array, reference_raster, out_path = None):
        
        '''
        save array as tif file. out path is whitout extention
        '''
        
        helper = ProcessingHelper(self.context, self.feedback)
                
        ref_path = reference_raster.dataProvider().dataSourceUri()
        ref_raster = gdal.Open(ref_path)
        # Create a temporary file
        if out_path != None:
            temp_file = QTemporaryFile(out_path)
        else:
            temp_file = QTemporaryFile('temp')
        
        # Open the file and get the file name
        temp_file.open()
        temp_raster_path = temp_file.fileName().split('.')[0] + '.tif'
        temp_file.close()
        
        print(f'file TEMP path: {temp_raster_path}')
        # Create the raster file writer
        writer = QgsRasterFileWriter(temp_raster_path)
        helper.saverasternd(ref_raster, temp_raster_path, array)
                
        return temp_raster_path         


    def reclassify_raster_searchsort(self, array, to_classes, actual_classes, nodata = 0): 
        '''
        array is the array to be mapped with aggr_ classes
        to_classes are a list of classes to put in place of actual_classes classes, 
        in this case it contains the nodata in the first position
        actual_classes is the list of classes that are present in the array,
        each calss will be remapped in to_classes,
        note that nodata value has to be passed searately adn will be mapped with
        the first position of to_classes list. 
        
        '''
        
        
        # add codification for no data:0
        
        _aggr = to_classes.copy()
        _codes = actual_classes.copy()
        
        # aggr_ is vulnerabilities, includes value for nodata in first position
        # so add no data class for array in the list of classes
        # _aggr.extend([nodata])
        _codes.insert(0, nodata) # (position, value)
    
        # nodata in codes will be mapped with this code: -1
        _aggr.insert(0, -1) # (position, value)
    
        # convert numpy array
        codes_np = np.array(_codes)
        aggr_np = np.array(_aggr)
        
        print('reclassification of array')
        print(f'array codes: {np.unique(array)}')
        print(f'input codes: {codes_np}')
        print(f'new codes: {aggr_np}')
        # check all values in the raster are present in the array of classes
        try:
            assert np.unique( [i in codes_np for i in np.unique(array)] ) # it will throw an error if expression is not valid
        except ValueError:
            # self.feedback.pushInfo('')
            print('ERROR: while remapping your array, some input codes where not covered.' +
                  ' this means there are codes in the input array that differs from your list, results will be incorrect')
            
        # do the mapping with fancy indexing --> very fast 
        # the idea is to link index of codes, position of code in array, and indices of codes to replace
        sort_idx = np.argsort(codes_np)
        idx = np.searchsorted(codes_np, array, sorter = sort_idx) # put codes indices in input array
        
        # create an array with indices of new classes linked to indices of input codes, 
        # then in the previous array of indeces put such new clases
        mapped_array = aggr_np[sort_idx][idx] 
        
        print(f'list of classes of reclassified array: {np.unique(mapped_array)}\n')
        
        return mapped_array


        






