import settings
from loader.model_loader import loadmodel
from feature_operation import hook_feature, FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean
import time
import activate_experiment
from places365_val_exp import Places365FeatureExtractor
import places365_val_exp

start_time = time.time()

fo = FeatureOperator()
pfe = Places365FeatureExtractor()
model = loadmodel(hook_feature)
model2 = loadmodel(places365_val_exp.hook_feature)

############ STEP 1: feature extraction ###############
features, maxfeature = fo.feature_extraction(model=model)

########### STEP 1.5: places365_val featre extraction ###########
places365_features = pfe.feature_extraction(model=model2)

for layer_id, layer in enumerate(settings.FEATURE_NAMES):
    ############ STEP 2: calculating threshold ############
    # thresholds.shape = (512,), where 512 is the number of layer-4 feature maps
    # calculating the threshold for each feature map in the layer-4
    thresholds = fo.quantile_threshold(features[layer_id], savepath="quantile.npy")

    ############ STEP 3: calculating IoU scores ###########
    tally_result = fo.tally(features[layer_id], thresholds, savepath="tally.csv")

    ############ STEP 4: generating results ###############
    rendered_order = generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)

    ########### STEP5: experiment about whether images activate more on units which have meanings ########
    # activate_experiment.activate_exp(fo.data, features[layer_id], thresholds, rendered_order)

    if settings.CLEAN:
        clean()

end_time = time.time()

m, s = divmod(end_time - start_time, 60)
h, m = divmod(m, 60)
print('Totally time: %02d:%02d:%02d' % (h, m, s))
