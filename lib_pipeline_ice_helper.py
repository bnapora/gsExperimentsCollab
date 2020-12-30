import numpy as np
from pathlib import Path
import openslide
from random import randint
from matplotlib import *
from tqdm import tqdm

from icevision.all import *
from icevision.parsers.mixins import *

# from object_detection_fastai.helper.object_detection_helper import *
# from object_detection_fastai.helper.wsi_loader import *

# from fastai import *
# from fastai.vision import *
# from fastai.callbacks import *
# from fastai.data_block import *

from lib_pipeline_ice_config import *

#Function modified from Object_Detection_Fastai - wsi_loader.py (https://github.com/ChristianMarzahl/ObjectDetection)
class SlideContainerIce():

    def __init__(self, file: Path, y, level: int=0, width: int=256, height: int=256, sample_func: callable=None):
        self.file = file
        self.slide = openslide.open_slide(str(file))
        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]
        self.y = y
        self.sample_func = sample_func
        self.classes = list(set(self.y[1]))

        if level is None:
            level = self.slide.level_count - 1
        self.level = level

    def get_patch(self,  x: int=0, y: int=0):
        # return np.array(self.slide.read_region(location=(int(x * self.down_factor),int(y * self.down_factor)),
                                        #   level=self.level, size=(self.width, self.height)))[:, :, :3]
        return self.slide.read_region(location=(int(x * self.down_factor),int(y * self.down_factor)), level=self.level, size=(self.width, self.height))

    @property
    def shape(self):
        return (self.width, self.height)

    def __str__(self):
        return str(self.file)

    def get_new_train_coordinates(self):
        # print('SlideContainer-get_new_train_coordinates()=',self)
        # use passed sampling method
        if callable(self.sample_func):
            return self.sample_func(self.y, **{"classes": self.classes, "size": self.shape,
                                               "level_dimensions": self.slide.level_dimensions,
                                               "level": self.level})

        # use default sampling method
        width, height = self.slide.level_dimensions[self.level]
        if len(self.y[0]) == 0:
            return randint(0, width - self.shape[0]), randint(0, height - self.shape[1])
        else:
            # use default sampling method
            class_id = np.random.choice( self.classes, 1)[0]
            ids = np.array( self.y[1]) == class_id

            bbox_coord = np.array( self.y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]
            xmin, ymin, _, _ = bbox_coord

            xmin, ymin = max(1, int(xmin - self.shape[0] / 2)), max(1, int(ymin - self.shape[1] / 2))
            xmin, ymin = min(xmin, width - self.shape[0]), min(ymin, height - self.shape[1])

            return xmin, ymin

    def get_patch_bboxeslabels(self, x: int=0, y: int=0):
        h, w = self.shape
        bboxes, labels = self.y

        bboxes = np.array([box for box in bboxes]) if len(np.array(bboxes).shape) == 1 else  np.array(bboxes)
        labels = np.array(labels)

        if len(labels) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - y

            bb_widths = (bboxes[:, 2] - bboxes[:, 0]) / 2
            bb_heights = (bboxes[:, 3] - bboxes[:, 1]) / 2

            ids = ((bboxes[:, 0] + bb_widths) > 0) \
                      & ((bboxes[:, 1] + bb_heights) > 0) \
                      & ((bboxes[:, 2] - bb_widths) < w) \
                      & ((bboxes[:, 3] - bb_heights) < h)
            bboxes = bboxes[ids]
            bboxes = np.clip(bboxes, 0, max(h,w))
            bboxes = bboxes[:, [1, 0, 3, 2]]
            labels = labels[ids]
        
        if len(labels) == 0:
            labels = np.array([0])
            bboxes = np.array([[0, 0, 1, 1]])
        wsicoords=[x,y]
        return wsicoords, bboxes, labels #, self.classes, self.pad_idx

#Function used to extract patches from slide and pass to SlideContainer()
def get_slide_annotations(database,patch_size,query_slides):

    files = []
    lbl_bbox = []

    for currslide, filename in tqdm(database.execute(query_slides).fetchall()):
        database.loadIntoMemory(currslide)

        slide_path = pathWSI / filename

        slide = openslide.open_slide(str(slide_path))
        level = 0#slide.level_count - 1
        level_dimension = slide.level_dimensions[level]
        down_factor = slide.level_downsamples[level]

        labels, bboxes = [], []

        for id, annotation in database.annotations.items():
            if annotation.labels[0].classId in classes:
                d = 2 * anno_radius / down_factor
                x_min = (annotation.x1 - anno_radius) / down_factor
                y_min = (annotation.y1 - anno_radius) / down_factor

                x_max = x_min + d
                y_max = y_min + d
                label = annotation.labels[0].classId

                bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                labels.append(label)

        if len(bboxes) > 0:
            lbl_bbox.append([bboxes, labels])

            files.append(SlideContainerIce(slide_path, y=[bboxes, labels],  level=level, width=patch_size, height=patch_size))
    return files, lbl_bbox

class FilePatchToMemoryRecordMixin(RecordMixin):
    def set_filepath(self, filepath: Union[str, Path], imgobject: None, imgcoords: Tuple[int, int]):
        self.filepath = Path(filepath)
        self.imgobject = imgobject
        self.imgcoords = imgcoords
    
    def _load(self):
        orig_coords = str(orig_anno_coord(self.imgcoords[0])) + '_' + str(orig_anno_coord(self.imgcoords[1]))

        roi_np = np.array(self.imgobject.get_patch(self.imgcoords[0],self.imgcoords[1]))
        self.img = roi_np[:, :, :3]
        self.height, self.width, _ = self.img.shape
        super()._load()

    def _autofix(self) -> Dict[str, bool]:
        exists = self.filepath.exists()
        if not exists:
            raise AutofixAbort(f"File '{self.filepath}' does not exist")

        return super()._autofix()

    def _repr(self) -> List[str]:
        return [f"Filepath: {self.filepath}", *super()._repr()]

    def as_dict(self) -> dict:
        # HACK: img, height, width are conditonal, use __dict__ to circumvent that
        # can be resolved once `as_dict` gets removed
        return {**self.__dict__, **super().as_dict()}

class FilePatchToMemoryMixin(ParserMixin):
    """Adds `filepath,imgobject,imgcoords` method to parser"""

    def record_mixins(self):
        return [FilePatchToMemoryRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        filepath = Path(self.filepath(o))
        if not filepath.exists():
            raise AbortParseRecord(f"File '{filepath}' does not exist")

        imgobject = self.imgobject(o)
        imgcoords = self.imgcoords(o)

        record.set_filepath(filepath,imgobject, imgcoords)
        super().parse_fields(o, record)

    @abstractmethod
    def filepath(self, o) -> Union[str, Path]:
        pass
    def imgobject(self, o) -> None:
        pass
    def imgcoords(self, o) -> Tuple[int, int]:
        pass

class SlideContainerParser(parsers.FasterRCNN, FilePatchToMemoryMixin, parsers.SizeMixin, parsers.LabelsMixin, parsers.BBoxesMixin):
    def __init__(self, slides):
        # self.source = source
        self.slides = slides
        self.patch_size = params['size']

    def __iter__(self):
        for patch in self.slides:
            self.patch_level = patch.level
            self.patch_x, self.patch_y = patch.get_new_train_coordinates()
            self.patch_wsicoords, self.patch_bboxes, self.patch_labels = patch.get_patch_bboxeslabels(self.patch_x,self.patch_y)
            yield patch
   
    def __len__(self):
        return len(self.slides)

    def imageid(self, o) -> Hashable:
        imageid = str(orig_anno_coord(self.patch_wsicoords[0])) + '_' + str(orig_anno_coord(self.patch_wsicoords[1]))
        return imageid

    def filepath(self, o) -> Union[str, Path]:
        filepath = o.file
        return filepath

    def imgobject(self, o) -> None:
        return o

    def imgcoords(self, o) -> Tuple[int, int]:
        return [self.patch_x,self.patch_y]

    def image_width_height(self, o) -> Tuple[int, int]:
        return [self.patch_size,self.patch_size]

    def labels(self, o) -> List[int]:
        labels = []
        for i, label in enumerate(self.patch_labels):
            labels.append(label)
        return labels

    def bboxes(self, o) -> List[BBox]:
        bboxes = []
        for i, bbox in enumerate(self.patch_bboxes):
            bbox = bb_hw(bbox)
            bbox = BBox.from_xywh(bbox[0],bbox[1],bbox[2],bbox[3])
            bboxes.append(bbox)
        return bboxes

######Misc Helper Functions#######################
#Function to calculate Annotation Position from SlideContainer Anno x & y
def ann_pos(coord,size,r):
    bb1 = coord + (size/2) + r
    return int(bb1)

#Used to determine original annotation coord
def orig_anno_coord (coord):
    coord = ann_pos(coord, size, anno_radius) 
    return coord

def draw_text1(ax, xy, txt, sz=14):
    xy[1] = xy[1] - 10
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline1(text, 1)

def draw_rect1(ax, x, y):
    patch = ax.add_patch(patches.Rectangle((x,y), 50,50, fill=False, edgecolor='white', lw=2))
    draw_outline1(patch, 4)

def draw_outline1(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_text(ax, xy, txt, sz=14):
     x = xy[0]
     y = xy[1]
     text = ax.text(x,y, txt,
        verticalalignment='baseline', color='white', fontsize=sz, weight='bold')
     draw_outline(text, 1)

#FastAI Internal - convert VOC's height/width into top-left/bottom-right, and switch x/y coords to be consistent with numpy.
def hw_bb(bb): return np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])

#Convert back to matplotlib graphical form
def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])

def print_properties(self):
    for prop, value in vars(self).items():
        print(prop, ":", value) # or use forma



