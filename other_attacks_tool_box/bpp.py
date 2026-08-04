from other_attacks_tool_box import BackdoorAttack
from utils import supervisor, tools
from other_attacks_tool_box.tools import generate_dataloader
import config
import copy
import torch
import numpy as np
import random
from tqdm import tqdm
import os


class attacker(BackdoorAttack):

    def __init__(
        self, args, mode="all2one", dithering=True, squeeze_num=8, injection_rate=0.2
    ):
        super().__init__(args)

        self.args = args
        self.train_source = supervisor.normalize_train_source(args.train_source)
        self.mode = mode
        self.dithering = dithering
        self.squeeze_num = squeeze_num
        self.injection_rate = injection_rate
        self.poison_transform = supervisor.get_poison_transform(
            poison_type=args.poison_type,
            dataset_name=args.dataset,
            target_class=config.target_class[args.dataset],
            trigger_transform=self.data_transform,
            is_normalized_input=True,
            alpha=args.alpha if args.test_alpha is None else args.test_alpha,
            trigger_name=args.trigger,
            args=args,
        )
        poison_set_dir = supervisor.get_poison_set_dir(args)
        if not os.path.exists(poison_set_dir):
            os.makedirs(poison_set_dir)
        if args.dataset == "cifar10":

            num_classes = 10
            arch = supervisor.get_arch(args)
            momentum = 0.9
            weight_decay = 1e-4
            epochs = 100
            milestones = torch.tensor([50, 75])
            learning_rate = 0.1
            batch_size = 512

        elif args.dataset == "cifar100":

            num_classes = 100
            arch = supervisor.get_arch(args)
            momentum = 0.9
            weight_decay = 1e-4
            epochs = 100
            milestones = torch.tensor([])
            learning_rate = 0.1
            batch_size = 512

        elif args.dataset == "tinyimagenet":

            num_classes = 200
            arch = supervisor.get_arch(args)
            momentum = 0.9
            weight_decay = 1e-4
            epochs = 100
            milestones = torch.tensor([])
            learning_rate = 0.1
            batch_size = 256

        elif args.dataset == "stl10":

            num_classes = 10
            arch = supervisor.get_arch(args)
            momentum = 0.9
            weight_decay = 1e-4
            epochs = 100
            milestones = torch.tensor([50, 75])
            learning_rate = 0.1
            batch_size = 128

        elif args.dataset == "imagenet100":

            num_classes = 100
            arch = supervisor.get_arch(args)
            momentum = 0.9
            weight_decay = 1e-4
            epochs = 100
            milestones = torch.tensor([])
            learning_rate = 0.1
            batch_size = 128

        elif args.dataset == "gtsrb":

            num_classes = 43
            arch = supervisor.get_arch(args)
            momentum = 0.9
            weight_decay = 1e-4
            epochs = 100
            milestones = torch.tensor([30, 60])
            learning_rate = 0.01
            batch_size = 128

        elif args.dataset == "imagenette":

            num_classes = 10
            arch = supervisor.get_arch(args)
            momentum = 0.9
            weight_decay = 1e-4
            epochs = 100
            milestones = torch.tensor([40, 80])
            learning_rate = 0.1
            batch_size = 128

        elif args.dataset == "imagenet":

            num_classes = 1000
            arch = supervisor.get_arch(args)
            momentum = 0.9
            weight_decay = 1e-4
            epochs = 90
            milestones = torch.tensor([30, 60])
            learning_rate = 0.1
            batch_size = 256
        else:
            raise NotImplementedError()

        # the if/elif branches above set local variables; bind the ones the base
        # class does not provide (epochs/milestones/batch_size) back onto self,
        # and refresh the rest so per-dataset overrides take effect.
        self.num_classes = num_classes
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.milestones = milestones
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        test_split = "test" if self.train_source == "test" else "full_test"
        self.train_loader = generate_dataloader(
            dataset=self.dataset,
            dataset_path=config.data_dir,
            batch_size=self.batch_size,
            split="train",
            shuffle=True,
            drop_last=False,
            data_transform=self.data_transform_aug,
            train_source=self.train_source,
            clean_budget=args.clean_budget,
            data_rate=args.data_rate,
        )
        self.test_loader = generate_dataloader(
            dataset=self.dataset,
            dataset_path=config.data_dir,
            batch_size=100,
            split=test_split,
            shuffle=False,
            drop_last=False,
            data_transform=self.data_transform,
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.milestones, 0.1
        )
        self.criterion_CE = torch.nn.CrossEntropyLoss()
        self.criterion_BCE = torch.nn.BCELoss()

        self.folder_path = "other_attacks_tool_box/results/bpp"
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def back_to_img(self, data):
        return self.denormalizer(data) * 255

    def img_tensor_norm(self, data):
        return self.normalizer(data / 255.0)

    def attack(self):
        self.model.to(self.device)
        residual_list_train = []
        residual_tag = (
            f"{self.args.dataset}_train_source={self.train_source}"
            f"_data_rate={self.args.data_rate}"
            f"_clean_budget={self.args.clean_budget}"
        )
        save_path = os.path.join(
            self.folder_path, f"{residual_tag}_residual_list_train"
        )
        if os.path.exists(save_path):
            residual_list_train = torch.load(save_path, map_location=self.device)
        else:
            for _ in range(1):
                for inputs, targets in tqdm(self.train_loader):
                    inputs = inputs.to(self.device)
                    temp_negetive = self.back_to_img(inputs)

                    temp_negetive_modified = copy.deepcopy(temp_negetive)
                    if self.dithering:
                        temp_negetive_modified = torch.round(
                            floydDitherspeed(
                                temp_negetive_modified, float(self.squeeze_num)
                            )
                        )
                    else:
                        temp_negetive_modified = (
                            torch.round(
                                temp_negetive_modified / 255.0 * (self.squeeze_num - 1)
                            )
                            / (self.squeeze_num - 1)
                            * 255
                        )

                    residual = temp_negetive_modified - temp_negetive
                    for i in range(residual.shape[0]):
                        residual_list_train.append(
                            residual[i].unsqueeze(0).to(self.device)
                        )
            torch.save(residual_list_train, save_path)
        for epoch in range(self.epochs):
            self.model.train()
            total_loss_ce = 0
            total_sample = 0
            total_clean = 0
            total_bd = 0
            total_clean_correct = 0
            total_bd_correct = 0

            for inputs, targets in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                bs = inputs.shape[0]
                num_bd = int(bs * self.injection_rate)
                num_neg = int(bs * self.injection_rate)
                inputs_bd = self.back_to_img(inputs[:num_bd])
                if self.dithering:
                    inputs_bd = torch.round(floydDitherspeed(inputs_bd, float(self.squeeze_num)))
                else:
                    inputs_bd = (
                        torch.round(inputs_bd / 255.0 * (self.squeeze_num - 1))
                        / (self.squeeze_num - 1)
                        * 255
                    )

                inputs_bd = self.img_tensor_norm(inputs_bd)

                if self.mode == "all2one":
                    targets_bd = torch.ones_like(targets[:num_bd]) * self.target_class
                if self.mode == "all2all":
                    targets_bd = torch.remainder(targets[:num_bd] + 1, self.num_classes)

                inputs_negative = self.back_to_img(
                    inputs[num_bd : (num_bd + num_neg)]
                ) + torch.cat(random.sample(residual_list_train, num_neg), dim=0)
                inputs_negative = torch.clamp(inputs_negative, 0, 255)
                inputs_negative = self.img_tensor_norm(inputs_negative)

                total_inputs = torch.cat(
                    [inputs_bd, inputs_negative, inputs[(num_bd + num_neg) :]], dim=0
                )
                total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

                # total_inputs = self.img_tensor_norm(total_inputs)
                total_preds = self.model(total_inputs)
                loss_ce = self.criterion_CE(total_preds, total_targets)
                loss = loss_ce
                loss.backward()
                self.optimizer.step()

                total_bd += num_bd
                total_sample += bs
                total_loss_ce += loss_ce.detach()
                total_clean += bs - num_bd - num_neg
                total_clean_correct += torch.sum(
                    torch.argmax(total_preds[(num_bd + num_neg) :], dim=1)
                    == total_targets[(num_bd + num_neg) :]
                )
                total_bd_correct += torch.sum(
                    torch.argmax(total_preds[:num_bd], dim=1) == targets_bd
                )
                avg_acc_bd = total_bd_correct * 100.0 / total_bd
                avg_acc_clean = total_clean_correct * 100.0 / total_clean

                # print("Epoch {} - Batch {}: Clean - {}, Backdoor - {}".format(epoch + 1, batch_idx, avg_acc_clean,
                #                                                                 avg_acc_bd))
            print("Epoch {}: Loss: {}".format(epoch + 1, loss))

            self.scheduler.step()

            if epoch % 1 == 0:
                tools.test(
                    model=self.model,
                    test_loader=self.test_loader,
                    poison_test=True,
                    poison_transform=self.poison_transform,
                    num_classes=self.num_classes,
                )
                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                torch.save(
                    model_to_save.state_dict(), supervisor.get_model_dir(self.args)
                )

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(), supervisor.get_model_dir(self.args))


def rnd1(x, decimals, out):
    # return np.round_(x, decimals, out)
    return torch.round(x.to(torch.double), decimals=decimals, out=out)


# def floydDitherspeed(image, squeeze_num):
#     channel, h, w = image.shape
#     for y in range(h):
#         for x in range(w):
#             old = image[:, y, x]
#             # temp = np.empty_like(old).astype(np.float64)
#             temp = torch.empty_like(old).to(torch.double).cuda()
#             new = rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
#             error = old - new
#             image[:, y, x] = new
#             if x + 1 < w:
#                 image[:, y, x + 1] += error * 0.4375
#             if (y + 1 < h) and (x + 1 < w):
#                 image[:, y + 1, x + 1] += error * 0.0625
#             if y + 1 < h:
#                 image[:, y + 1, x] += error * 0.3125
#             if (x - 1 >= 0) and (y + 1 < h):
#                 image[:, y + 1, x - 1] += error * 0.1875
#     return image


def floydDitherspeed(image, squeeze_num):
    bs, c, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:, :, y, x]
            # temp = np.empty_like(old).astype(np.float64)
            temp = torch.empty_like(old, dtype=torch.double, device=image.device)
            new = (
                rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
            )
            error = old - new
            image[:, :, y, x] = new
            if x + 1 < w:
                image[:, :, y, x + 1] += error * 0.4375
            if (y + 1 < h) and (x + 1 < w):
                image[:, :, y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[:, :, y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h):
                image[:, :, y + 1, x - 1] += error * 0.1875
    return image


class poison_transform:
    def __init__(
        self,
        img_size,
        normalizer,
        denormalizer,
        mode="all2one",
        dithering=True,
        squeeze_num=8,
        num_classes=10,
        target_class=0,
    ):
        self.img_size = img_size
        self.normalizer = normalizer
        self.denormalizer = denormalizer
        self.mode = mode
        self.dithering = dithering
        self.squeeze_num = squeeze_num
        self.num_classes = num_classes
        self.target_class = target_class  # by default : target_class = 0

    def transform(self, data, labels):
        data = data.clone()
        labels = labels.clone()
        # transform clean samples to poison samples

        labels[:] = self.target_class

        data = self.denormalizer(data) * 255
        if self.dithering:
            data = torch.round(floydDitherspeed(data, float(self.squeeze_num)))
        else:
            data = (
                torch.round(data / 255.0 * (self.squeeze_num - 1))
                / (self.squeeze_num - 1)
                * 255
            )
        data = self.normalizer(data / 255.0)

        if self.mode == "all2one":
            labels = torch.ones_like(labels) * self.target_class
        if self.mode == "all2all":
            labels = torch.remainder(labels + 1, self.num_classes)

        from torchvision.utils import save_image

        # save_image(self.denormalizer(data), "a.png")
        # exit()

        return data, labels
