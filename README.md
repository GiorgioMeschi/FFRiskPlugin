# FFRiskPlugin

A QGIS plugin for wildfire risk assessment at national/regional level.

Three modules are present:

1) Wildfire Hazard Plugin: evaluation of wildifire susceptibility and hazard with machine learning modelling 

2) Processing Risk Plugin: data preparation for the risk assessment (creation of single files for several exposed elements, identified as point of interest such as schools and  hospitals, roads and vegetation types such as croplands) 

3) Wildfire Risk Plugin: assessment of element-specific risk maps and total risk at national/regional level. Hazard and exposed elements are linked through empirical vulnerability curves which define a relation between the intensity of a wildfire and the degree of damage of the asset, thus defining a potential impact which can be also an economic estimation if monetary values are associated with each element at risk.  