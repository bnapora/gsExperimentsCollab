from pathlib import Path
from datetime import *
import pytz

params = dict(
    size=512,
    size_trans=512,
    size_progresize=0,
    batch_size=2,
    ds_size=100,
    run='1',
    architecture='CNN',
    modelclass='EffDet',
    infrastructure='unknown',
    )
params.update(project= 'sample_project')
params['model'] = 'tf_efficientdet_d4'
params['run_datetime'] = datetime.now(pytz.timezone('US/Pacific')).strftime("%m%d%Y-%H:%M")

anno_radius = 25 
params['bbox_hw'] = anno_radius

classes = {2:  'mitotic figure'}
class_map_list = ['unkown','mitotic figure']

params.update(class_map=class_map_list)

size = params['size']  
path = Path('./')

dbname = 'MITOS_WSI_CCMCT_ODAEL.sqlite'

#Standalone Paths
path_WSI = '/home/bnapora/development/gsExperimentsCollab/WSI/'
path_Database = '/home/bnapora/development/gsExperimentsCollab/WSI/'

pathWSI = Path(path_WSI)
pathDB = Path(path_Database)

# slidelist_test = ['30']
slidelist_test= '''('30','29')'''

params['dbname'] = dbname