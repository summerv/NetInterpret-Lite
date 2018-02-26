import os
import settings
import numpy
from scipy.misc import imresize

UNITS_COUNT = 512

def activate_exp(data, features, thresholds, rendered_order):
    '''
    Check whether activated units are meaningful units(having concept)
    :param data:
    :param features:
    :param thresholds:
    :param rendered_order:
    :return:
    '''
    fnidxmap = data.fnidxmap
    with open(os.path.join(settings.DATA_DIRECTORY, '..', 'activate_exp_data_layer3.txt'), 'r') as f:
        img_fn_list = [line.strip() for line in f.readlines()]
    prefix = 'dataset/broden1_224/images/'

    # activate_rate[img][unit] == x, img activates x(percent) pixels on this unit
    activate_rate = numpy.zeros((len(img_fn_list), UNITS_COUNT), dtype='float')
    for i, img_fn in enumerate(img_fn_list):
        img_idx = fnidxmap[img_fn.replace(prefix, '')]
        print(img_fn.replace(prefix, ''))
        for unit in range(UNITS_COUNT):    # layer4 has 512 units
            mask = imresize(features[img_idx][unit], (settings.IMG_SIZE, settings.IMG_SIZE), mode='F')
            mask = mask > thresholds[unit]
            activate_rate[i][unit] = numpy.count_nonzero(mask) / pow(settings.IMG_SIZE, 2)
    print(activate_rate.shape)

    # sorted by unit index (start from 1)
    rendered_order = sorted(rendered_order, key=lambda record: (int(record['unit'])))

    activate_info = [None] * len(img_fn_list)
    for img_idx, img_acti in enumerate(activate_rate):
        img_info = [None] * UNITS_COUNT    # each img_info contains UNITS_COUNT {}, each {} includes unit/score/cat/label
        for unit_idx, unit_acti in enumerate(img_acti):
            unit_info = {
                'unit': unit_idx,
                'score': rendered_order[unit_idx]['score'],
                'label': rendered_order[unit_idx]['label'],
                'category': rendered_order[unit_idx]['category'],
                'acti_rate': unit_acti
            }
            img_info[unit_idx] = unit_info
        img_info_sorted = sorted(img_info, key=lambda record: (-float(record['acti_rate']), -float(record['score'])))
        activate_info[img_idx] = img_info_sorted

    return activate_info


if __name__ == '__main__':
    pass