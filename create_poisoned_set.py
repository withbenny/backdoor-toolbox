import os
import torch
from torchvision import datasets, transforms
import argparse
from PIL import Image
import numpy as np
import config
from utils import supervisor, default_args, tools
from torch.utils.data import Subset

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str,  required=False,
                    choices=default_args.parser_choices['poison_type'],
                    default=default_args.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
parser.add_argument('-data_rate', type=float,  required=True, default=1.0)
args = parser.parse_args()

tools.setup_seed(0)


print('[target class : %d]' % config.target_class[args.dataset])

data_dir = config.data_dir  # directory to save standard clean set
if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]

if not os.path.exists(os.path.join('poisoned_train_set', args.dataset)):
    os.mkdir(os.path.join('poisoned_train_set', args.dataset))

def reduce_data(dataset, data_rate):
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, '_labels'):
        targets = np.array(dataset._labels)
    elif hasattr(dataset, 'samples'):
        targets = np.array([s[1] for s in dataset.samples])
    else:
        print("Using iterative method to extract labels from dataset.")
        try:
            targets = np.array([label for _, label in dataset])
        except Exception as e:
            raise ValueError("Unable to extract labels from dataset. Please check the dataset format.") from e
    if len(targets) == 0:
        raise ValueError("The dataset is empty or does not contain labels.")

    unique_classes, counts = np.unique(targets, return_counts=True)
    original_counts = dict(zip(unique_classes, counts))
    reduced_indices = []
    reduced_counts = {}

    print(f"Reducing dataset to {data_rate * 100}% of original size...")
    for cls in unique_classes:
        class_indices = np.where(targets == cls)[0]
        original_count = len(class_indices)

        num_to_keep = int(np.round(original_count * data_rate))
        if num_to_keep == 0 and data_rate > 0 and original_count > 0:
            num_to_keep = 1
        
        num_to_keep = min(num_to_keep, original_count)

        np.random.shuffle(class_indices)
        selected_indices = class_indices[:num_to_keep]
        reduced_indices.extend(selected_indices.tolist())
        reduced_counts[cls] = len(selected_indices)
    
    np.random.shuffle(reduced_indices)
    print(f"Oringinal counts: {original_counts}, Reduced counts: {reduced_counts}")
    
    return Subset(dataset, reduced_indices)

if args.poison_type == 'dynamic':

    if args.dataset == 'cifar10':

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                     download=True, transform=data_transform)
        img_size = 32
        num_classes = 10
        channel_init = 32
        steps = 3
        input_channel = 3

        ckpt_path = './models/all2one_cifar10_ckpt.pth.tar'

        normalizer = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])

        denormalizer = transforms.Compose([
            transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261], [1 / 0.247, 1 / 0.243, 1 / 0.261])
        ])

    elif args.dataset == 'gtsrb':

        data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split='train',
                                   transform=data_transform, download=True)

        img_size = 32
        num_classes = 43
        channel_init = 32
        steps = 3
        input_channel = 3

        ckpt_path = './models/all2one_gtsrb_ckpt.pth.tar'

        normalizer = None
        denormalizer = None

    elif args.dataset == 'imagenette':
        raise  NotImplementedError('imagenette unsupported for dynamic!')
    else:
        raise  NotImplementedError('Undefined Dataset')

elif args.poison_type == 'ISSBA':

    if args.dataset == 'cifar10':

        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                     download=True, transform=data_transform)
        img_size = 32
        num_classes = 10
        input_channel = 3

        ckpt_path = './models/ISSBA_cifar10.pth'

    elif args.dataset == 'gtsrb':

        data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split='train',
                                   transform=data_transform, download=True)

        img_size = 32
        num_classes = 43
        input_channel = 3

        ckpt_path = './models/ISSBA_gtsrb.pth'

    elif args.dataset == 'imagenette':
        raise  NotImplementedError('imagenette unsupported!')
    else:
        raise  NotImplementedError('Undefined Dataset')

else:

    if args.dataset == 'gtsrb':

        data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split = 'train',
                                   transform = data_transform, download=True)
        img_size = 32
        num_classes = 43

    elif args.dataset == 'cifar10':

        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                     download=True, transform=data_transform)
        img_size = 32
        num_classes = 10
    
    elif args.dataset == 'stl10':

        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.STL10(os.path.join(data_dir, 'stl10'), split='train',
                                   download=True, transform=data_transform)
        img_size = 96
        num_classes = 10

    elif args.dataset == 'imagenette':

        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train_set = datasets.ImageFolder(os.path.join(os.path.join(data_dir, 'imagenette2'), 'train'),
                                         data_transform)
        img_size = 224
        num_classes = 10

    else:
        raise  NotImplementedError('Undefined Dataset')

if args.data_rate < 1.0:
    train_set = reduce_data(train_set, args.data_rate)

trigger_transform = transforms.Compose([
    transforms.ToTensor()
])

# Create poisoned dataset directory for current setting
poison_set_dir = supervisor.get_poison_set_dir(args)
# poison_set_img_dir = os.path.join(poison_set_dir, 'data')

if os.path.exists(poison_set_dir):
    print(f"Poisoned set directory '{poison_set_dir}' to be created is not empty! Exiting...")
    exit()
if not os.path.exists(poison_set_dir):
    os.mkdir(poison_set_dir)
# if not os.path.exists(poison_set_img_dir):
#     os.mkdir(poison_set_img_dir)



if args.poison_type in ['basic', 'badnet', 'blend', 'clean_label', 'refool',
                        'adaptive_blend', 'adaptive_patch', 'adaptive_k_way',
                        'SIG', 'TaCT', 'WaNet', 'SleeperAgent', 'none',
                        'badnet_all_to_all', 'trojan']:

    trigger_name = args.trigger
    trigger_path = os.path.join(config.triggers_dir, trigger_name)

    trigger = None
    trigger_mask = None

    if trigger_name != 'none':  # none for SIG

        print('trigger: %s' % trigger_path)

        trigger_path = os.path.join(config.triggers_dir, trigger_name)
        trigger = Image.open(trigger_path).convert("RGB")
        trigger = trigger_transform(trigger)

        trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_name)
        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            #print('trigger_mask_path:', trigger_mask_path)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            #print('No trigger mask found! By default masking all black pixels...')
            trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0), trigger[2] > 0).float()

    alpha = args.alpha

    poison_generator = None
    if args.poison_type == 'basic':

        from poison_tool_box import basic
        poison_generator = basic.poison_generator(img_size=img_size, dataset=train_set,
                                                  poison_rate=args.poison_rate,
                                                  path=poison_set_dir,
                                                  trigger_mark=trigger, trigger_mask=trigger_mask,
                                                  target_class=config.target_class[args.dataset], alpha=alpha)
        
    elif args.poison_type == 'badnet':

        from poison_tool_box import badnet
        poison_generator = badnet.poison_generator(img_size=img_size, dataset=train_set,
                                                   poison_rate=args.poison_rate, trigger_mark=trigger, trigger_mask=trigger_mask,
                                                   path=poison_set_dir, target_class=config.target_class[args.dataset])

    elif args.poison_type == 'badnet_all_to_all':

        from poison_tool_box import badnet_all_to_all
        poison_generator = badnet_all_to_all.poison_generator(img_size=img_size, dataset=train_set,
                                                   poison_rate=args.poison_rate, trigger_mark=trigger, trigger_mask=trigger_mask,
                                                   path=poison_set_dir, num_classes=num_classes)

    elif args.poison_type == 'trojan':

        from poison_tool_box import trojan
        poison_generator = trojan.poison_generator(img_size=img_size, dataset=train_set,
                                                 poison_rate=args.poison_rate, trigger_mark=trigger, trigger_mask=trigger_mask,
                                                 path=poison_set_dir, target_class=config.target_class[args.dataset])

    elif args.poison_type == 'blend':

        from poison_tool_box import blend
        poison_generator = blend.poison_generator(img_size=img_size, dataset=train_set,
                                                  poison_rate=args.poison_rate, trigger=trigger,
                                                  path=poison_set_dir, target_class=config.target_class[args.dataset],
                                                  alpha=alpha)
    elif args.poison_type == 'refool':
        from poison_tool_box import refool
        poison_generator = refool.poison_generator(img_size=img_size, dataset=train_set,
                                                  poison_rate=args.poison_rate,
                                                  path=poison_set_dir, target_class=config.target_class[args.dataset],
                                                  max_image_size=32)

    elif args.poison_type == 'TaCT':

        from poison_tool_box import TaCT
        poison_generator = TaCT.poison_generator(img_size=img_size, dataset=train_set,
                                                 poison_rate=args.poison_rate, cover_rate=args.cover_rate,
                                                 trigger=trigger, mask=trigger_mask,
                                                 path=poison_set_dir, target_class=config.target_class[args.dataset],
                                                 source_class=config.source_class,
                                                 cover_classes=config.cover_classes)

    elif args.poison_type == 'WaNet':
        # Prepare grid
        s = 0.5
        k = 4
        grid_rescale = 1
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            torch.nn.functional.upsample(ins, size=img_size, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d = torch.linspace(-1, 1, steps=img_size)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        
        path = os.path.join(poison_set_dir, 'identity_grid')
        torch.save(identity_grid, path)
        path = os.path.join(poison_set_dir, 'noise_grid')
        torch.save(noise_grid, path)

        from poison_tool_box import WaNet
        poison_generator = WaNet.poison_generator(img_size=img_size, dataset=train_set,
                                                 poison_rate=args.poison_rate, cover_rate=args.cover_rate,
                                                 path=poison_set_dir,
                                                 identity_grid=identity_grid, noise_grid=noise_grid,
                                                 s=s, k=k, grid_rescale=grid_rescale, 
                                                 target_class=config.target_class[args.dataset])

    elif args.poison_type == 'adaptive':

        from poison_tool_box import adaptive
        poison_generator = adaptive.poison_generator(img_size=img_size, dataset=train_set,
                                                     poison_rate=args.poison_rate,
                                                     path=poison_set_dir,
                                                     trigger_mark=trigger, trigger_mask=trigger_mask,
                                                     target_class=config.target_class[args.dataset], alpha=alpha,
                                                     cover_rate=args.cover_rate)
    
    elif args.poison_type == 'adaptive_blend':

        from poison_tool_box import adaptive_blend
        poison_generator = adaptive_blend.poison_generator(img_size=img_size, dataset=train_set,
                                                          poison_rate=args.poison_rate,
                                                          path=poison_set_dir, trigger=trigger,
                                                          pieces=16, mask_rate=0.5,
                                                          target_class=config.target_class[args.dataset], alpha=alpha,
                                                          cover_rate=args.cover_rate)
    
    elif args.poison_type == 'adaptive_patch':


        from poison_tool_box import adaptive_patch
        poison_generator = adaptive_patch.poison_generator(img_size=img_size, dataset=train_set,
                                                           poison_rate=args.poison_rate,
                                                           path=poison_set_dir,
                                                           trigger_names=config.adaptive_patch_train_trigger_names[args.dataset],
                                                           alphas=config.adaptive_patch_train_trigger_alphas[args.dataset],
                                                           target_class=config.target_class[args.dataset],
                                                           cover_rate=args.cover_rate)

    elif args.poison_type == 'adaptive_k_way':

        from poison_tool_box import adaptive_k_way
        poison_generator = adaptive_k_way.poison_generator(img_size=img_size, dataset=train_set,
                                                           poison_rate=args.poison_rate,
                                                           path=poison_set_dir,
                                                           target_class=config.target_class[args.dataset],
                                                           cover_rate=args.cover_rate)

    elif args.poison_type == 'SIG':

        from poison_tool_box import SIG
        poison_generator = SIG.poison_generator(img_size=img_size, dataset=train_set,
                                                poison_rate=args.poison_rate,
                                                path=poison_set_dir, target_class=config.target_class[args.dataset],
                                                delta=30/255, f=6)

    elif args.poison_type == 'clean_label':

        if args.dataset == 'cifar10':
            adv_imgs_path = "data/cifar10/clean_label/fully_poisoned_training_datasets/two_600.npy"
            if not os.path.exists("data/cifar10/clean_label/fully_poisoned_training_datasets/two_600.npy"):
                raise NotImplementedError("Run 'data/cifar10/clean_label/setup.sh' first to launch clean label attack!")
            adv_imgs_src = np.load("data/cifar10/clean_label/fully_poisoned_training_datasets/two_600.npy").astype(
                np.uint8)
            adv_imgs = []
            for i in range(adv_imgs_src.shape[0]):
                adv_imgs.append(data_transform(adv_imgs_src[i]).unsqueeze(0))
            adv_imgs = torch.cat(adv_imgs, dim=0)
            assert adv_imgs.shape[0] == len(train_set)
        else:
            raise NotImplementedError('Clean Label Attack is not implemented for %s' % args.dataset)

        # Init Attacker
        from poison_tool_box import clean_label
        poison_generator = clean_label.poison_generator(img_size=img_size, dataset=train_set, adv_imgs=adv_imgs,
                                                        poison_rate=args.poison_rate,
                                                        trigger_mark = trigger, trigger_mask=trigger_mask,
                                                        path=poison_set_dir, target_class=config.target_class[args.dataset])

    elif args.poison_type == 'SleeperAgent':
        from poison_tool_box import SleeperAgent
        
        if args.dataset == 'cifar10':
            normalizer = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])

            denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261], [1 / 0.247, 1 / 0.243, 1 / 0.261])
            ])
            
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
            ])
            
            trainset = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                        download=True, transform=data_transform)
            testset = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=False,
                                       download=True, transform=data_transform)
        else: raise(NotImplementedError)
        poison_generator = SleeperAgent.poison_generator(img_size=img_size, model_arch=supervisor.get_arch(args),
                                                         random_patch=False,
                                                         dataset=trainset, testset=testset,
                                                         poison_rate=args.poison_rate, path=poison_set_dir,
                                                         normalizer=normalizer, denormalizer=denormalizer,
                                                         source_class=config.source_class,
                                                         target_class=config.target_class[args.dataset])
    
    else: # 'none'
        from poison_tool_box import none
        poison_generator = none.poison_generator(img_size=img_size, dataset=train_set,
                                                path=poison_set_dir)



    if args.poison_type not in ['TaCT', 'WaNet', 'adaptive_blend', 'adaptive_patch', 'adaptive_k_way']:
        img_set, poison_indices, label_set = poison_generator.generate_poisoned_training_set()
        print('[Generate Poisoned Set] Save %d Images' % len(label_set))

    else:
        img_set, poison_indices, cover_indices, label_set = poison_generator.generate_poisoned_training_set()
        print('[Generate Poisoned Set] Save %d Images' % len(label_set))

        cover_indices_path = os.path.join(poison_set_dir, 'cover_indices')
        torch.save(cover_indices, cover_indices_path)
        print('[Generate Poisoned Set] Save %s' % cover_indices_path)


    img_path = os.path.join(poison_set_dir, 'imgs')
    torch.save(img_set, img_path)
    print('[Generate Poisoned Set] Save %s' % img_path)

    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)

    #print('poison_indices : ', poison_indices)


elif args.poison_type == 'dynamic':
    """
        Since we will use the pretrained model by the original paper, here we use normalized data following 
        the original implementation.
        Download Pretrained Generator from https://github.com/VinAIResearch/input-aware-backdoor-attack-release
    """
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('[Dynamic Attack] Download pretrained generator first : https://github.com/VinAIResearch/input-aware-backdoor-attack-release')
    # Init Attacker
    from poison_tool_box import dynamic
    poison_generator = dynamic.poison_generator(ckpt_path=ckpt_path, channel_init=channel_init, steps=steps,
                                                input_channel=input_channel, normalizer=normalizer,
                                                denormalizer=denormalizer, dataset=train_set,
                                                poison_rate=args.poison_rate, path=poison_set_dir, target_class=config.target_class[args.dataset])

    # Generate Poison Data
    img_set, poison_indices, label_set = poison_generator.generate_poisoned_training_set()
    print('[Generate Poisoned Set] Save %d Images' % len(label_set))

    img_path = os.path.join(poison_set_dir, 'imgs')
    torch.save(img_set, img_path)
    print('[Generate Poisoned Set] Save %s' % img_path)
    
    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)

elif args.poison_type == 'ISSBA':
    # if not os.path.exists(ckpt_path):
    #     raise NotImplementedError('[ISSBA Attack] Download pretrained encoder and decoder first: https://github.com/')
    
    # Init Secret
    secret_size = 20
    secret = torch.FloatTensor(np.random.binomial(1, .5, secret_size).tolist())
    secret_path = os.path.join(poison_set_dir, 'secret')
    torch.save(secret, secret_path)
    print('[Generate Poisoned Set] Save %s' % secret_path)
    
    # Init Attacker
    from poison_tool_box import ISSBA
    poison_generator = ISSBA.poison_generator(ckpt_path=ckpt_path, secret=secret, dataset=train_set, enc_height=img_size, enc_width=img_size, enc_in_channel=input_channel,
                                                poison_rate=args.poison_rate, path=poison_set_dir, target_class=config.target_class[args.dataset])

    # Generate Poison Data
    img_set, poison_indices, label_set = poison_generator.generate_poisoned_training_set()
    print('[Generate Poisoned Set] Save %d Images' % len(label_set))

    img_path = os.path.join(poison_set_dir, 'imgs')
    torch.save(img_set, img_path)
    print('[Generate Poisoned Set] Save %s' % img_path)
    
    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)

else:
    raise NotImplementedError('%s not defined' % args.poison_type)