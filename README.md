# Description

A QGIS plugin for wildfire risk assessment at national/regional level.
By default the risk plug-in works with OpenStreetMap data but the user can input custom data.

Three modules are present:

1) Wildfire Hazard Plugin: evaluation of wildifire susceptibility and hazard with machine learning modelling 

2) Processing Risk Plugin: data preparation for the risk assessment (creation of single files for several exposed elements, identified as point of interest such as schools and  hospitals, roads and vegetation types such as croplands) 

3) Wildfire Risk Plugin: assessment of element-specific risk maps and total risk at national/regional level. Hazard and exposed elements are linked through empirical vulnerability curves (editable by the user) which define empirical relations between the intensity of a wildfire and the degree of damage on the asset, thus defining a potential impact which can be either an economic estimation (if monetary values are provided) or a qualitative value. 

# Versioning

Two sets of plug ins are available:

veriosn_ipaff: it contains the 3 modules and a tutorial file corresponding to the development for the EU-funded project IPAFF. 

AAL_version: it contains the 3 modules and a tutorial file, the main differences are: 
1. The hazard map is defined as 12 classes and not 6 as in the IPAFF verison. 
2. A likelihood map associated with the hazard classes defining how much, on average, each class is expected to burn (in percentage) is computed
3. The risk computation is more accurate and goes toward an estimation of Average Annual Losses of the selected exposed elements.
4. The user has to pass in input average economic values for the assets (healthcare facilities, industrial, etc)

# Installation 

Please refer on the following version of QGIS.
Download the QGIS 3.16.15-1 (Hannover, long term release) at the following website:
https://download.qgis.org/downloads/
make sure to download the complete package, searching for the file QGIS-OSGeo4W-3.16.15-1.msi
which release date is 2021-12-18 and the size is 1 Gb. 

The 3 modules (WildfireHazardPlugin, ProcessingRiskPlugin and WildfireRiskPlugin) have to be dowloaded from the current repository,
zipped and loaded in QGIS: Plugin -> Manage and install plugins.. -> Install from ZIP (from the sidebar) -> drag and drop each zip file

Please note that scikit-learn may need a manual installation.
If an error occurs during the installation of the WildfireHazardPlugin, follow the subsequent steps:

1)	Open the OSGeo4w Shell searching for it in the application search bar of your computer. 
2)	Type the following command and press enter: pip install scikit-learn
3)	Come back to QGIS, (in Plugins | installed sidebar remove manually the RFForestFireRisk module) 
4)  Proceed again with the plug in installation
5)  Open the processing toolbox (processing -> toolbox) and search for the installed modules on the right sidebar

# Tutorial

View the file tutorial.docx  and follow the steps using the data available in data_for_testing folder to reproduce the
risk maps. The risk module has 2 sections depending on the version used (IPAFF or AAL).
The user can use its own data as long as they follow the specifics of the one provided.

# Potential issues

1) manual installation of scikit-learn throught OSGeo4w Shell
2) the elevetion layer's data type (DEM file) has to be float 
3) fully integration for windows OS, further tests have to be conducted on Unix-like OS (potential minor bugs may occur)





