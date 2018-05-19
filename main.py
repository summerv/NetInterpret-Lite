import settings
from loader.model_loader import loadmodel
from feature_operation import hook_feature, FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean
import time
import activate_experiment
import places365_val_exp
from places365_val_exp import Places365FeatureExtractor

start_time = time.time()

fo = FeatureOperator()
pfe = Places365FeatureExtractor()
model = loadmodel(hook_feature)
model2 = loadmodel(places365_val_exp.hook_feature)

# ----- STEP 1: feature extraction-----
ext_time = time.time()
features, maxfeature = fo.feature_extraction(model=model)
print("extract broden feature using time: %f s.", (time.time() - ext_time))

# -----STEP 1.5: places365_val featre
ext_time2 = time.time()
places365_features, true_index, images_365, predicted_label = pfe.feature_extraction(model=model2)
print("extract place365 feature using time: %f s.", (time.time() - ext_time2))

for layer_id, layer in enumerate(settings.FEATURE_NAMES):
    # ----- STEP 2: calculating threshold -----
    # thresholds.shape = (512,), where 512 is the number of layer-4 feature maps
    # calculating the threshold for each feature map in the layer-4
    thresh_time = time.time()
    thresholds = fo.quantile_threshold(features[layer_id], savepath="quantile.npy")
    print("compute threshold using time: %f s.", (time.time() - thresh_time))

    # -----STEP 3: calculating IoU scores-----
    iou_time = time.time()
    tally_result = fo.tally(features[layer_id], thresholds, savepath="tally.csv")
    print("compute threshold using time: %f s.", (time.time() - iou_time))

    # -----STEP 4: generating results-----
    rendered_order = generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)

    # -----STEP5: experiment about whether images activate more on units which have meanings-----
    # activate_experiment.activate_exp_broden(fo.data, features[layer_id], thresholds, rendered_order)
    dt_fea_time = time.time()
    activate_rate, dt_features, true_images, feature_names = activate_experiment.activate_exp_places365(images_365, true_index, places365_features[layer_id], thresholds, rendered_order)
    # activate_experiment.activate_exp_places365(images_365, true_index, places365_features[layer_id], thresholds, rendered_order)
    print("dt features computing using time: %f s.", (time.time() - dt_fea_time))
    '''
    tree_time = time.time()
    activate_experiment.gen_gbdt(dt_features, true_images, feature_names)
    print("tree computing using time: %f s.", (time.time() - tree_time))
    '''
    selected_true_images, target_names, feature_names = activate_experiment.gen_sample_activate_rate(activate_rate, true_images, true_index, feature_names)
    activate_experiment.feature_location(selected_true_images, target_names, features[layer_id], thresholds, feature_names)

    # data for wlw
    # activate_experiment.get_featuremap(features[layer_id], selected_true_images)
    import pickle
    import os
    in_file = open(os.path.join("feature_map_5class.pkl"), "rb")

    select_featuremap = pickle.load(in_file)
    print(select_featuremap)
    print(select_featuremap.shape)

    if settings.CLEAN:
        clean()

end_time = time.time()

m, s = divmod(end_time - start_time, 60)
h, m = divmod(m, 60)
print('Totally time: %02d:%02d:%02d' % (h, m, s))
