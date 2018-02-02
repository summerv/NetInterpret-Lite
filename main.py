import settings
from loader.model_loader import loadmodel
from feature_operation import hook_feature, FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean

fo = FeatureOperator()
model = loadmodel(hook_feature)

############ STEP main.py:91: feature extraction ###############
features, maxfeature = fo.feature_extraction(model=model)

for layer_id, layer in enumerate(settings.FEATURE_NAMES):
    ############ 32STEP 2: calculating threshold ############
    # thresholds.shape = (512,), where 512 is the number of layer-4 feature maps
    # calculating the threshold for each feature map in the layer-4
    thresholds = fo.quantile_threshold(features[layer_id], savepath="quantile.npy")

    ############ STEP 3: calculating IoU scores ###########
    tally_result = fo.tally(features[layer_id], thresholds, savepath="tally.csv")

    ############ STEP 4: generating results ###############
    generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)
    if settings.CLEAN:
        clean()
