from qgis.core import QgsProcessing, QgsField, QgsProcessingUtils, QgsVectorFileWriter, QgsRasterFileWriter, QgsVectorLayer, QgsProject,QgsCoordinateReferenceSystem
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
        
        return path, raster, array

        
        
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


    def veg_aggregation_str(self, vector_layer, veg_arr, name_veg_col = 'veg', name_intens_col = 'aggr'):
        
        codes = list()
        aggregation = list()
        
        for feature in vector_layer.getFeatures():
            codes.append(feature[name_veg_col])
            aggregation.append(feature[name_intens_col])
            
        codes = [int(i) for i in codes] 
        codes = [str(i) for i in codes] 
        aggregation = [str(i) for i in aggregation]
        
        print('veg codes:\n', codes, 'intensity:\n', aggregation)
        
        veg_arr_aggr = veg_arr.astype(int)
        veg_arr_aggr = veg_arr_aggr.astype(str)

        
        breaking = False
        for i in aggregation:
            if i in codes:
                self.feedback.pushInfo('the vegetation codes must be different from aggregation codes')
                breaking = True
                break
            else:
                pass
            
        if not breaking:    
            for i,j in zip(codes, aggregation):         
                veg_arr_aggr = np.where(veg_arr_aggr == i, j, veg_arr_aggr)
        else:
           self.feedback.pushInfo('could not procede, fix your vegetation codes') 
        
        
        return veg_arr_aggr


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


    def raster_buffer(self, raster_to_buffer, buffer_window):
        
        # clip borders
        raster_no_borders = raster_to_buffer[1:raster_to_buffer.shape[0]-1,1:raster_to_buffer.shape[1]-1].astype(np.int8())
        raster_mask_border = np.zeros(raster_to_buffer.shape)
        raster_mask_border[1:raster_to_buffer.shape[0]-1,1:raster_to_buffer.shape[1]-1] = raster_no_borders
        # dilatation
        structure = np.ones((buffer_window, buffer_window))
        buffered_mask = ndimage.morphology.binary_dilation(raster_mask_border == 1, structure = structure)  # , structure=createKernel(1))   
        buffered_img_noBorder = np.where(buffered_mask == True, 1, 0).astype(np.int8())
        buffered_img = raster_to_buffer.copy().astype(np.int8())
        
        # add again border values inserting original raster values 
        buffered_img[1:buffered_img.shape[0]-1,1:buffered_img.shape[1]-1] = buffered_img_noBorder[1:buffered_img.shape[0]-1,1:buffered_img.shape[1]-1].astype(np.int8())
        
        return buffered_img


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
        
        print(f'list of classes of reclassified array: {np.unique(mapped_array)}')
        
        return mapped_array


    
    def raster_classes_vals(self, data, bounds, nodata, norm = False):
        '''    
        bounds define the values for the classes, extreme values included.
        first class starts from first value of bounds, included, and end with last one. 
        '''
        
        data3 = np.where(data == nodata, np.NaN, data)
        del data
        if norm == True:
            data3 = data3/np.nanmax(data3)
        else:
            pass
        
        # convert the raster map into a categorical map based on  values
        conditions = list()
        for i in range(0, len(bounds)-1 ):
            # first position take also lowlimit value, the others dont include it
            if i == 0:
                conditions.append( ((data3 >= bounds[i]) & (data3 <= bounds[i+1])) )
            else:
                conditions.append( ((data3 > bounds[i]) & (data3 <= bounds[i+1])) )
    
        classes = [i for i in range(1, len(bounds))]
        
        out_arr = np.select(conditions, classes, default=0)
        out_arr = out_arr.astype(np.int8())
        
        print(f'classes mapped array: {np.unique(out_arr)}')
        
        return out_arr


    def duplicate_point_layer(self, layer, crs):
        
        # duplicate the layer in order to be sure it has all the correct metadata 
        feats = [feat for feat in layer.getFeatures()]

        mem_layer = QgsVectorLayer(f"Point?crs={crs}", "duplicated_layer", "memory")
        
        mem_layer_data = mem_layer.dataProvider()
        attr = layer.dataProvider().fields().toList()
        mem_layer_data.addAttributes(attr)
        mem_layer.updateFields()
        mem_layer_data.addFeatures(feats)
        
        return mem_layer

    def duplicate_linear_layer(self, layer, crs):
        
        # duplicate the layer in order to be sure it has all the correct metadata 
        feats = [feat for feat in layer.getFeatures()]

        mem_layer = QgsVectorLayer(f"LineString?crs={crs}", "duplicated_layer_road", "memory")
        
        mem_layer_data = mem_layer.dataProvider()
        attr = layer.dataProvider().fields().toList()
        mem_layer_data.addAttributes(attr)
        mem_layer.updateFields()
        mem_layer_data.addFeatures(feats)
        
        return mem_layer

        

    def save_shapefile(self, reference_layer, out_shp_path, crs):
        
                            
        _crs = QgsCoordinateReferenceSystem(crs)
        # Save the layer to a shapefile
        error = QgsVectorFileWriter.writeAsVectorFormat(reference_layer, out_shp_path, 'UTF-8', _crs, driverName = 'ESRI Shapefile')
        
        # Check for errors
        if error[0] != QgsVectorFileWriter.NoError:
            print(f"Error saving layer: {error}")
        else:
            print(f"Layer saved as {out_shp_path}")

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
        

    def reproject_vector_layer(self, layer, prj, to_path = None):
        
        if to_path is not None:
            parameter = {
                'INPUT': layer,
                'TARGET_CRS': prj,
                'OUTPUT': to_path
            }
        else:
            parameter = {
                'INPUT': layer,
                'TARGET_CRS': prj,
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
            }
        
        layer_res = processing.run('native:reprojectlayer', parameter) 
        
        return layer_res




