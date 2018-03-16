import os
import settings
import numpy
from scipy.misc import imresize
from sklearn import tree
import graphviz
from matplotlib.image import imread
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

UNITS_COUNT = 512

def activate_exp_broden(data, features, thresholds, rendered_order):
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
        img_class_list = [line.strip()[1] for line in f.readlines()]

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

def activate_exp_places365(images, true_index, features, thresholds, rendered_order):
    '''
    Check whether activated units are meaningful units(having concept)
    :param images:
    :param true_index:
    :param features:
    :param thresholds:
    :param rendered_order:
    :return: activate_info, dt_features, true_images, feature_names
    '''

    print("start features calculating...")
    # activate_info = []
    dt_features = None    # dt_features.shape = (len(true_images), UNITS_COUNT)
    dt_features_size = None
    true_images = []  # only include images classified correctly, [{'img_idx':xx(0~36500), 'img_fn':xx, 'img_label':xx(0-355)}]
    feature_names = []

    # activate_info_file = os.path.join(settings.PLACES365_VAL_OUTPUT_FOLDER, "true_activate_info.npy")
    dt_features_file = os.path.join(settings.PLACES365_VAL_OUTPUT_FOLDER, "dt_features_%s.mmap" % settings.FEATURE_NAMES[0])
    dt_features_size_file = os.path.join(settings.PLACES365_VAL_OUTPUT_FOLDER, "dt_features_size.npy")
    true_images_file = os.path.join(settings.PLACES365_VAL_OUTPUT_FOLDER, "true_images.npy")
    feature_names_file = os.path.join(settings.PLACES365_VAL_OUTPUT_FOLDER, "feature_names.npy")

    skip = True
    if os.path.exists(dt_features_size_file) and os.path.exists(true_images_file) and os.path.exists(feature_names_file):
        dt_features_size = numpy.load(dt_features_size_file)
        true_images = numpy.load(true_images_file)
        feature_names = numpy.load(feature_names_file)
    else:
        skip = False

    if os.path.exists(dt_features_file) and dt_features_size is not None:
        print('loading dt_features...')
        dt_features = numpy.memmap(dt_features_file, dtype=float, mode='r', shape=tuple(dt_features_size))
    else:
        print("dt_features file missing, loading from scratch")
        skip = False

    if skip:
        return dt_features, true_images, feature_names

    # activate_rate[img][unit] == x, img activates x(percent) pixels on this unit
    activate_rate = numpy.zeros((len(images), UNITS_COUNT), dtype='float')

    for img_idx, img in enumerate(images):
        if img_idx % 100 == 0:
            print("processing img_idx: %d / %d" % (img_idx, len(images)))
        if not true_index[img_idx]:
            continue
        img_fn = img['image']
        img_label = img['label']
        true_images.append({'img_idx': img_idx, 'img_fn': img_fn, 'img_label': img_label})
        for unit in range(UNITS_COUNT):    # layer4 has 512 units
            mask = imresize(features[img_idx][unit], (settings.IMG_SIZE, settings.IMG_SIZE), mode='F')
            mask = mask > thresholds[unit]
            activate_rate[img_idx][unit] = numpy.count_nonzero(mask) / pow(settings.IMG_SIZE, 2)
    print('activate.shape: ', activate_rate.shape)

    # sorted by unit index (start from 1)
    rendered_order = sorted(rendered_order, key=lambda record: (int(record['unit'])))

    for i, item in enumerate(rendered_order):
        feature_names.append(item['label'])

    for idx, img in enumerate(true_images):
        # img_info = [None] * UNITS_COUNT       # each img_info contains UNITS_COUNT {}, each {} includes unit/score/cat/label
        img_features = [None] * UNITS_COUNT   # 512 features in dic format
        for unit_idx, unit_acti in enumerate(activate_rate[img['img_idx']]):
            # unit_info = {
            #     'unit': unit_idx,
            #     'score': rendered_order[unit_idx]['score'],
            #     'label': rendered_order[unit_idx]['label'],
            #     'category': rendered_order[unit_idx]['category'],
            #     'acti_rpate': unit_acti
            # }
            # img_info[unit_idx] = unit_info
            img_features[unit_idx] = float(rendered_order[unit_idx]['score']) * unit_acti

        # img_info_sorted = sorted(img_info, key=lambda record: (-float(record['acti_rate']), -float(record['score'])))
        # activate_info.append(img_info_sorted)

        if dt_features is None:
            dt_features_size = (len(true_images), UNITS_COUNT)
            dt_features = numpy.memmap(dt_features_file, dtype=float, mode='w+', shape=dt_features_size)

        dt_features[idx] = img_features

    numpy.save(dt_features_size_file, dt_features_size)
    print(numpy.array(dt_features_size).shape)
    numpy.save(feature_names_file, feature_names)
    print(numpy.array(feature_names).shape)
    numpy.save(true_images_file, true_images)
    print(numpy.array(true_images).shape)
    print("dt_features_size: ", dt_features_size)

    # numpy.save(activate_info_file, activate_info)
    # print(numpy.array(activate_info).shape)
    print("finishing features calculating...")

    return dt_features, true_images, feature_names


def gen_gbdt(dt_features, true_images, feature_names):
    with open(os.path.join(settings.PLACES365_VAL_DIRECTORY, 'categories_places365.txt')) as f:
        target_names = numpy.array([line.split(' ')[0].split('/')[2:] for line in f.readlines()])

    for name in target_names:
        if len(name) > 1:
            name[0] += '/' + name[1]

    target_names = [name[0] for name in target_names]

    statistic = {}  # {label: count} in 'true_images'
    for img in true_images:
        statistic[target_names[img['img_label']]] = statistic.setdefault(target_names[img['img_label']], 0) + 1
    statistic_sorted = sorted(statistic.items(), key=lambda d: d[1], reverse=False)

    print('------ statistic {label: count} in "true_images" ------')
    for item in statistic_sorted:
        print(item)

    # selected_label_list = [item[0] for item in statistic if item[1] >= 90]
    # selected_label_list = ['beach', 'bedroom', 'bookstore', 'waterfall', 'swimming_pool/indoor']
    selected_label_list = ['bamboo_forest']
    for label in selected_label_list:
        print(label, " count:", statistic[label])
    print(selected_label_list)

    selected_true_images_flag = [True if target_names[img['img_label']] in selected_label_list else False for img in true_images ]

    selected_dt_features = [feature for i, feature in enumerate(dt_features) if selected_true_images_flag[i]]

    # -------- draw the heatmap of dt_features to see whether the same class would activate same units ---------
    df = pd.DataFrame(numpy.array(selected_dt_features))
    # df = pd.DataFrame(selected_dt_features, columns=[x for x in feature_names])
    sns.heatmap(df, annot=False)
    plt.show()

    x = numpy.array(selected_dt_features)
    labels = numpy.array([img['img_label'] for i, img in enumerate(true_images) if selected_true_images_flag[i]])
    feature_details = [None] * len(x)
    print("len_label:", len(labels))
    print("dt_features.shape: ", dt_features.shape)
    for idx, img_features in enumerate(selected_dt_features):
        feature_detail = {}
        for i in range(len(img_features)):
            if img_features[i] > 0.0:
                feature_detail[feature_names[i]] = img_features[i]
        feature_detail[target_names[labels[idx]]] = 100    # label_name
        feature_detail = sorted(feature_detail.items(), key=lambda d: d[1], reverse=True)
        feature_details[idx] = feature_detail
        # img = imread(os.path.join(settings.PLACES365_VAL_DIRECTORY, 'images_224', true_images[idx]['img_fn']))
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        # print("feature_details[%d]:" % idx)
        # print(feature_details[idx])

    statistic = {}
    for img in true_images:
        if target_names[img['img_label']] in selected_label_list:
            statistic[target_names[img['img_label']]] = statistic.setdefault(target_names[img['img_label']], 0) + 1
    statistic = sorted(statistic.items(), key=lambda d: d[1], reverse=False)
    print(statistic)

    # clf = tree.DecisionTreeClassifier()
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=20,
                                      min_samples_leaf=10)  # CART, entropy as criterion
    clf = clf.fit(x, labels)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=selected_label_list,
                                    filled=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("test4")

2
if __name__ == '__main__':
    pass