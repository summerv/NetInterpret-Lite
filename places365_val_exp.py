import os
import settings
import re
import numpy as np
import torch
from torch.autograd import Variable as V
import matplotlib.pyplot as plt
# from matplotlib.image import imread
from scipy.misc import imresize, imread, imsave

features_blobs = []
BATCH_SIZE = 1


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


class Places365FeatureExtractor:

    def __init__(self):
        self.bgr_mean = [109.5388, 118.6897, 124.6901]

        if not os.path.exists(settings.PLACES365_VAL_OUTPUT_FOLDER):
            os.makedirs(os.path.join(settings.PLACES365_VAL_OUTPUT_FOLDER))

        with open(os.path.join(settings.PLACES365_VAL_DIRECTORY, 'categories_places365.txt')) as f:
            self.cats_map = [line.split(' ')[0].split('/')[2:] for line in f.readlines()]

        with open(os.path.join(settings.PLACES365_VAL_DIRECTORY, 'places365_val.txt')) as f:
            # images: [{'image':image_file_name, 'label':label} * image_size]
            self.images = [self.decode_label_dict(line) for line in f.readlines()]

    def normalize_image(self, rgb_image, bgr_mean):
        """
            Load input image and preprocess for Caffe:
            - cast to float
            - switch channels RGB -> BGR
            - subtract mean
            - transpose to channel x height x width order
            """
        img = np.array(rgb_image, dtype=np.float32)
        if (img.ndim == 2):
            img = np.repeat(img[:, :, None], 3, axis=2)
        img = img[:, :, ::-1]
        if bgr_mean is not None:
            img -= bgr_mean
        img = img.transpose((2, 0, 1))
        return img

    def feature_extraction(self, model=None, memmap=True):
        wholefeatures = [None] * len(settings.FEATURE_NAMES)
        features_size = [None] * len(settings.FEATURE_NAMES)
        features_size_file = os.path.join(settings.PLACES365_VAL_OUTPUT_FOLDER, "places365val_feature_size.npy")
        trueidx_file = os.path.join(settings.PLACES365_VAL_OUTPUT_FOLDER, "true_index.npy")
        predicted_label = [None] * len(self.images)
        predicted_label_file = os.path.join(settings.PLACES365_VAL_OUTPUT_FOLDER, "places365val_predicted_label.npy")
        true_count = 0.0    # count number of images which are predicted correctly
        true_index = [None] * 36500     # true_index[i]: i-th image is classified correctly or not

        if memmap:
            skip_size = True
            skip_trueidx = True
            mmap_files = [os.path.join(settings.PLACES365_VAL_OUTPUT_FOLDER, "%s.mmap" % feature_name) for feature_name in settings.FEATURE_NAMES]

            if os.path.exists(features_size_file):
                features_size = np.load(features_size_file)
            else:
                skip_size = False

            if os.path.exists(trueidx_file):
                true_index = np.load(trueidx_file)
            else:
                skip_trueidx = False

            for i, mmap_file in enumerate(mmap_files):
                if os.path.exists(mmap_file) and os.path.exists(trueidx_file) and features_size[i] is not None:
                    print('loading features %s' % settings.FEATURE_NAMES[i])
                    wholefeatures[i] = np.memmap(mmap_file, dtype=float, mode='r', shape=tuple(features_size[i]))
                else:
                    print('file missing, loading from scratch')
                    skip_size = False

            if skip_size and skip_trueidx:
                return wholefeatures, true_index, self.images

        num_batches = len(self.images)
        for batch_idx, batch in enumerate(self.images):
            # features_blobs is a cache for extracted features,
            # features_blobs.shape[0] == len(batch[0]) == the number of images in a batch == BATCH_SIZE,
            # features_blobs.shape[1:] -> e.g. (512, 7, 7) ->
            # (size of feature maps in the given layer, featuremap size, featuremap size).
            del features_blobs[:]

            input = imread(os.path.join(settings.PLACES365_VAL_DIRECTORY, 'images_224', batch['image']))
            input = self.normalize_image(input, self.bgr_mean)
            input = input[None, :]
            Y = batch['label']
            batch_size = 1
            print('extracting places365_val feature from batch %d / %d' % (batch_idx+1, num_batches))
            input = torch.from_numpy(input[:, ::-1, :, :].copy())
            input.div_(255.0 * 0.224)

            if settings.GPU:
                input = input.cuda()
            input_var = V(input, volatile=True)    # input_var.shape can be like(len(images in batch[0]), 3, 224, 224)
            logit = model.forward(input_var)       # input_var.shape can be like(len(images in batch[0]), 365#classes#)
            predicted_Y = np.argmax(logit[0].data.cpu().numpy())
            predicted_label[batch_idx] = predicted_Y

            if Y == predicted_Y:
                true_count += 1.0
                true_index[batch_idx] = True
            else:
                true_index[batch_idx] = False

            if batch_idx == 0:
                print(logit)

            while np.isnan(logit.data.max()):
                print("nan")                       # which I have no idea why it will happen
                del features_blobs[:]
                logit = model.forward(input_var)
                predicted_Y = np.argmax(logit[0])
                predicted_label[batch_idx] = predicted_Y

            feat_batch = features_blobs[0]
            if len(feat_batch.shape) == 4 and wholefeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    # e.g. wholefeatures[i].shape = (36500, 512, 7, 7)
                    # There are 36500 IMAGES in places365_val in total!!!
                    size_features = (len(self.images), feat_batch.shape[1], feat_batch.shape[2], feat_batch.shape[3])
                    features_size[i] = size_features
                    if memmap:
                        wholefeatures[i] = np.memmap(mmap_files[i], dtype=float, mode='w+', shape=size_features)
                    else:
                        wholefeatures[i] = np.zeros(size_features)
            np.save(features_size_file, features_size)
            np.save(trueidx_file, true_index)
            np.save(predicted_label_file, predicted_label)

            start_idx = batch_idx*BATCH_SIZE
            end_idx = min((batch_idx+1)*BATCH_SIZE, len(self.images))
            for i, feat_batch in enumerate(features_blobs):
                if len(feat_batch.shape) == 4:
                    wholefeatures[i][start_idx:end_idx] = feat_batch
        print(true_count)
        print("acc: %.2f" % (true_count / len(self.images)))   # 19303/36500 = 0.52
        return wholefeatures, true_index, self.images

    def preprocess_places365(self):
        '''
        resize places365_val into (settings.PLACES365_VAL_SIZE, settings.PLACES365_VAL_SIZE, 3)
        cats_map: cats_map[index] = [root_cat, sub_cat]
        images: images[index] = {'image': file_name; 'label': label}
        :return:
        '''
        with open(os.path.join(settings.PLACES365_VAL_DIRECTORY, 'categories_places365.txt')) as f:
            cats_map = [line.split(' ')[0].split('/')[2:] for line in f.readlines()]

        with open(os.path.join(settings.PLACES365_VAL_DIRECTORY, 'places365_val.txt')) as f:
            images = [self.decode_label_dict(line) for line in f.readlines()]
            for key, val in enumerate(images, 1):
                image = imread(os.path.join(settings.PLACES365_VAL_DIRECTORY, 'images_256', val['image']))
                # plt.imshow(image)
                # plt.axis('off')
                # plt.show()
                # print(cats_map[val['label']])
                resize_image = imresize(image, (settings.PLACES365_VAL_SIZE, settings.PLACES365_VAL_SIZE))
                imsave(os.path.join(settings.PLACES365_VAL_DIRECTORY, 'images_'+str(settings.PLACES365_VAL_SIZE),
                                    val['image']), resize_image)
        return cats_map, images

    def decode_label_dict(self, row):
        result = {}
        split_list = row.strip().split(' ')
        for split in split_list:
            if '.jpg' in split:
                result['image'] = split
            if re.match('[1-9]\d*|0', split):
                result['label'] = int(split)
        return result


if __name__ == '__main__':
    # preprocess_places365()
    x = imread(os.path.join(settings.PLACES365_VAL_DIRECTORY, 'images_224', 'Places365_val_00000001.jpg'))
    print(x.shape)
