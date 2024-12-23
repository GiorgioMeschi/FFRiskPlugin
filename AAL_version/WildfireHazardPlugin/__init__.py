# -*- coding: utf-8 -*-
"""
/***************************************************************************
 RFForestFireRisk
                                 A QGIS plugin
 Forest Fire Risk analysis using Random Forest Algorith
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2022-01-04
        copyright            : (C) 2022 by CIMA Research Foundation
        email                : mirko.dandrea@cimafoundation.org
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""

__author__ = 'CIMA Research Foundation'
__date__ = '2022-01-04'
__copyright__ = '(C) 2022 by CIMA Research Foundation'


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load RFForestFireRisk class from file RFForestFireRisk.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .plugin import WildfireHazardPlugin
    return WildfireHazardPlugin()
