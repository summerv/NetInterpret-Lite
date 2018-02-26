import matplotlib.pyplot as plt
from matplotlib.image import imread
from feature_operation import FeatureOperator
import settings
import os
import numpy

class ReadImageAttribute:
    '''
    Show the attributes of the image with the given file name
    '''
    def __init__(self):
        fo = FeatureOperator()
        self.data = fo.data
        self.image = self.data.image
        self.fnidxmap = self.data.fnidxmap

    def get_imgage_idx(self, file_name):
        return self.fnidxmap[file_name]

    def show_image(self, file_name):
        img = imread(os.path.join(settings.DATA_DIRECTORY, 'images', file_name))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def get_size(self, file_name):
        ih = self.image[self.fnidxmap[file_name]]['ih']
        iw = self.image[self.fnidxmap[file_name]]['iw']
        sh = self.image[self.fnidxmap[file_name]]['sh']
        sw = self.image[self.fnidxmap[file_name]]['sw']
        return ih, iw, sh, sw

    def get_color(self, file_name):
        '''
        :param file_name:
        :return: 3-D color array of the given image
        '''
        color_fns = self.image[self.fnidxmap[file_name]]['color']
        ih, iw, sh, sw = self.get_size(file_name)
        if not color_fns:
            color = numpy.empty((1, sh, sw), dtype=numpy.int16)
        else:
            color = numpy.empty((len(color_fns), sh, sw), dtype=numpy.int16)
            for i, color_fn in enumerate(color_fns):
                png = imread(os.path.join(settings.DATA_DIRECTORY, 'images', color_fn))
                color[i] = png[:, :, 0] + png[:, :, 1] * 256
        return color

    def get_object(self, file_name):
        '''
        :param file_name:
        :return: 3-D object array of the given image
        '''
        obj_fns = self.image[self.fnidxmap[file_name]]['object']
        ih, iw, sh, sw = self.get_size(file_name)
        if not obj_fns:
            object = numpy.empty((1, sh, sw), dtype=numpy.int16)
        else:
            object = numpy.empty((len(obj_fns), sh, sw), dtype=numpy.int16)
            for i, obj_fn in enumerate(obj_fns):
                png = imread(os.path.join(settings.DATA_DIRECTORY, 'images', obj_fn))
                object[i] = png[:, :, 0] + png[:, :, 1] * 256
        return object

    def get_part(self, file_name):
        '''
        :param file_name:
        :return: 3-D part array of the given image
        '''
        part_fns = self.image[self.fnidxmap[file_name]]['part']
        ih, iw, sh, sw = self.get_size(file_name)
        if not part_fns:
            part = numpy.empty((1, sh, sw), dtype=numpy.int16)
        else:
            part = numpy.empty((len(part_fns), sh, sw), dtype=numpy.int16)
            for i, part_fn in enumerate(part_fns):
                png = imread(os.path.join(settings.DATA_DIRECTORY, 'images', part_fn))
                part[i] = png[:, :, 0] + png[:, :, 1] * 256
        return part

    def get_material(self, file_name):
        '''
        :param file_name:
        :return: 3-D material array of the given image
        '''
        mat_fns = self.image[self.fnidxmap[file_name]]['material']
        ih, iw, sh, sw = self.get_size(file_name)
        if not mat_fns:
            mat = numpy.empty((1, sh, sw), dtype=numpy.int16)
        else:
            mat = numpy.empty((len(mat_fns), sh, sw), dtype=numpy.int16)
            for i, mat_fn in enumerate(mat_fns):
                png = imread(os.path.join(settings.DATA_DIRECTORY, 'images', mat_fn))
                mat[i] = png[:, :, 0] + png[:, :, 1] * 256
        return mat

    def get_scene(self, file_name):
        '''
        :param file_name:
        :return: scene list of the given image
        '''
        scene_ids = self.image[self.fnidxmap[file_name]]['scene']
        return scene_ids

    def get_texture(self, file_name):
        '''
        :param file_name:
        :return: texture list of the given image
        '''
        texture_ids = self.image[self.fnidxmap[file_name]]['texture']
        return texture_ids


if __name__ == '__main__':
    ria = ReadImageAttribute()
    file_name = 'ade20k/ADE_train_00004567.jpg'
    ria.show_image(file_name)
    print(ria.get_imgage_idx(file_name))
    color = ria.get_color(file_name)
    print('color shape', color.shape)
    mat = ria.get_material(file_name)
    print('mat shape', mat.shape)
    part = ria.get_part(file_name)
    print('part shape', part.shape)
    obj = ria.get_object(file_name)
    print('obj shape', obj.shape)
    scene = ria.get_scene(file_name)
    print('scene', scene)
    texture = ria.get_texture(file_name)
    print('texture', texture)