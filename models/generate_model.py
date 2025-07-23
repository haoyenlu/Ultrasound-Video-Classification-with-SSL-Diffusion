from models import resnet, resnext , wide_resnet , pre_act_resnet, densenet
import torch.nn as nn



def generate_model(model_depth,
                   model_name,
                   n_classes,
                   sample_size,
                   sample_duration,
                   resnet_shortcut = 'A',
                   wide_resnet_k = 2,
                   resnext_cardinality=32,
                   mode = 'score',
                   no_cuda = False):
    
    assert mode in ['score', 'feature'], "Mode should be 'score' or 'feature'"
    if mode == 'score':
        last_fc = True
    elif mode == 'feature':
        last_fc = False

    assert model_name in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'], "Model not support"

    if model_name == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]

        if model_depth == 10:
            model = resnet.resnet10(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                    sample_size=sample_size, sample_duration=sample_duration,
                                    last_fc=last_fc)
        elif model_depth == 18:
            model = resnet.resnet18(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                    sample_size=sample_size, sample_duration=sample_duration,
                                    last_fc=last_fc)
        elif model_depth == 34:
            model = resnet.resnet34(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                    sample_size=sample_size, sample_duration=sample_duration,
                                    last_fc=last_fc)
        elif model_depth == 50:
            model = resnet.resnet50(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                    sample_size=sample_size, sample_duration=sample_duration,
                                    last_fc=last_fc)
        elif model_depth == 101:
            model = resnet.resnet101(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                     sample_size=sample_size, sample_duration=sample_duration,
                                     last_fc=last_fc)
        elif model_depth == 152:
            model = resnet.resnet152(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                     sample_size=sample_size, sample_duration=sample_duration,
                                     last_fc=last_fc)
        elif model_depth == 200:
            model = resnet.resnet200(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                     sample_size=sample_size, sample_duration=sample_duration,
                                     last_fc=last_fc)
    elif model_name == 'wideresnet':
        assert model_depth in [50]

        if model_depth == 50:
            model = wide_resnet.resnet50(num_classes=n_classes, shortcut_type=resnet_shortcut, k=wide_resnet_k,
                                         sample_size=sample_size, sample_duration=sample_duration,
                                         last_fc=last_fc)
    elif model_name == 'resnext':
        assert model_depth in [50, 101, 152]

        if model_depth == 50:
            model = resnext.resnet50(num_classes=n_classes, shortcut_type=resnet_shortcut, cardinality=resnext_cardinality,
                                     sample_size=sample_size, sample_duration=sample_duration,
                                     last_fc=last_fc)
        elif model_depth == 101:
            model = resnext.resnet101(num_classes=n_classes, shortcut_type=resnet_shortcut, cardinality=resnext_cardinality,
                                      sample_size=sample_size, sample_duration=sample_duration,
                                      last_fc=last_fc)
        elif model_depth == 152:
            model = resnext.resnet152(num_classes=n_classes, shortcut_type=resnet_shortcut, cardinality=resnext_cardinality,
                                      sample_size=sample_size, sample_duration=sample_duration,
                                      last_fc=last_fc)
    elif model_name == 'preresnet':
        assert model_depth in [18, 34, 50, 101, 152, 200]

        if model_depth == 18:
            model = pre_act_resnet.resnet18(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                            sample_size=sample_size, sample_duration=sample_duration,
                                            last_fc=last_fc)
        elif model_depth == 34:
            model = pre_act_resnet.resnet34(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                            sample_size=sample_size, sample_duration=sample_duration,
                                            last_fc=last_fc)
        elif model_depth == 50:
            model = pre_act_resnet.resnet50(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                            sample_size=sample_size, sample_duration=sample_duration,
                                            last_fc=last_fc)
        elif model_depth == 101:
            model = pre_act_resnet.resnet101(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                             sample_size=sample_size, sample_duration=sample_duration,
                                             last_fc=last_fc)
        elif model_depth == 152:
            model = pre_act_resnet.resnet152(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                             sample_size=sample_size, sample_duration=sample_duration,
                                             last_fc=last_fc)
        elif model_depth == 200:
            model = pre_act_resnet.resnet200(num_classes=n_classes, shortcut_type=resnet_shortcut,
                                             sample_size=sample_size, sample_duration=sample_duration,
                                             last_fc=last_fc)
    elif model_name == 'densenet':
        assert model_depth in [121, 169, 201, 264]

        if model_depth == 121:
            model = densenet.densenet121(num_classes=n_classes,
                                         sample_size=sample_size, sample_duration=sample_duration,
                                         last_fc=last_fc)
        elif model_depth == 169:
            model = densenet.densenet169(num_classes=n_classes,
                                         sample_size=sample_size, sample_duration=sample_duration,
                                         last_fc=last_fc)
        elif model_depth == 201:
            model = densenet.densenet201(num_classes=n_classes,
                                         sample_size=sample_size, sample_duration=sample_duration,
                                         last_fc=last_fc)
        elif model_depth == 264:
            model = densenet.densenet264(num_classes=n_classes,
                                         sample_size=sample_size, sample_duration=sample_duration,
                                         last_fc=last_fc)

    if not no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    return model
