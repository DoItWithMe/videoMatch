import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import io

from .iscnet.isc_model import ISCNet
from torchvision import transforms
from PIL import Image
from loguru import logger as log
from torch.utils.data import DataLoader, Dataset
from .exception import exception_handler


class __IscNetDataSet(Dataset):
    def __init__(
        self,
        imgs_list,
        preprocessor,
    ):
        self.preprocessor = preprocessor
        self.dataset = imgs_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = Image.open(img_path).convert("RGB")
        img = self.preprocessor(img)

        imgs = list()
        imgs.append(img)

        return (
            torch.stack(imgs),
            index,
        )


def __flatten_data(samples, device):
    # batch_size = samples.size(0)
    # num_samples_per_image = samples.size(1)
    channels = samples.size(2)
    height = samples.size(3)
    width = samples.size(4)

    return samples.view(-1, channels, height, width).to(device)


@exception_handler
def create_isc_model(
    weight_file_path: str,
    fc_dim: int = 256,
    p: float = 1.0,
    eval_p: float = 1.0,
    l2_normalize: bool = True,
    device: str = "cuda",
    is_training: bool = False,
) -> tuple[nn.DataParallel[ISCNet] | ISCNet, transforms.Compose]:
    """create_isc_model _summary_

    Args:
        weight_file_path (str): _description_
        fc_dim (int, optional): _description_. Defaults to 256.
        p (float, optional): _description_. Defaults to 1.0.
        eval_p (float, optional): _description_. Defaults to 1.0.
        l2_normalize (bool, optional): _description_. Defaults to True.
        device (str, optional): _description_. Defaults to "cuda".
        is_training (bool, optional): _description_. Defaults to False.

    Returns:
        tuple[nn.DataParallel[ISCNet] | ISCNet, transforms.Compose]: _description_
    """
    if device == "cuda":
        ckpt = torch.load(weight_file_path)
    else:
        ckpt = torch.load(weight_file_path, map_location="cpu")

    arch = ckpt["arch"]  # tf_efficientnetv2_m_in21ft1k
    input_size = ckpt["args"].input_size

    if arch == "tf_efficientnetv2_m_in21ft1k":
        arch = "timm/tf_efficientnetv2_m.in21k_ft_in1k"

    backbone = timm.create_model(arch, features_only=True)
    model = ISCNet(
        backbone=backbone,
        fc_dim=fc_dim,
        p=p,
        eval_p=eval_p,
        l2_normalize=l2_normalize,
    )

    model.to(device).train(is_training)

    state_dict = {}
    for s in ckpt["state_dict"]:
        state_dict[s.replace("module.", "")] = ckpt["state_dict"][s]

    if fc_dim != 256:
        # interpolate to new fc_dim
        state_dict["fc.weight"] = (
            F.interpolate(
                state_dict["fc.weight"].permute(1, 0).unsqueeze(0),
                size=fc_dim,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 0)
        )
        for bn_param in ["bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"]:
            state_dict[bn_param] = (
                F.interpolate(
                    state_dict[bn_param].unsqueeze(0).unsqueeze(0),
                    size=fc_dim,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

    model.load_state_dict(state_dict)

    assert input_size == 512
    preprocessor = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=backbone.default_cfg["mean"],
                std=backbone.default_cfg["std"],
            ),
        ]
    )

    if device == "cuda":
        model = torch.nn.DataParallel(model.cuda())

    return model, preprocessor


@exception_handler
def gen_img_feats_by_ISCNet(
    imgs_list: list[io.BytesIO],
    model: nn.Module,
    preprocessor: transforms.Compose,
    device: str = "cuda",
):
    """gen_img_feats_by_ISCNet _summary_

    Args:
        imgs_path_list (list[str]): _description_
        model (nn.Module): _description_
        preprocessor (transforms.Compose): _description_
        device (str, optional): _description_. Defaults to "cuda".

    Raises:
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        _type_: _description_

    ISCNet Usage:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n
        image = Image.open(requests.get(url, stream=True).raw)\n
        x = preprocessor(image).unsqueeze(0)\n
        y = model(x)\n
        log.info(y.shape)  # => torch.Size([1, 256])\n
    """

    if isinstance(model, nn.DataParallel):
        if not isinstance(model.module, ISCNet):
            raise RuntimeError(f"unknown model: {type(model)} -- {type(model.module)}")
    else:
        if not isinstance(model, ISCNet):
            raise RuntimeError(f"unknown model: {type(model)}")

    if preprocessor is None:
        raise RuntimeError(f"processcessor is not set!")

    from utils.time_utils import TimeRecorder

    tmp_recorder = TimeRecorder()
    tmp_recorder2 = TimeRecorder()

    dataset = __IscNetDataSet(imgs_list, preprocessor)
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    log.info("FUKC 1?")
    feats_list = list()

    tmp_recorder.start_record()
    for _, (imgs, index) in enumerate(data_loader):
        log.info("FUKC 2???????????????????")
        flatten_imgs = __flatten_data(imgs, device)
        with torch.no_grad():
            tmp_recorder2.start_record()
            imgs_feat = model(flatten_imgs).detach().cpu().numpy()
            tmp_recorder2.end_record()

            for img_feat in imgs_feat:
                feats_list.append(img_feat.reshape(-1))
    tmp_recorder.end_record()

    feats_array = np.array(feats_list)  # type: ignore

    log.info(
        f"extract feats with device: {device}, batch size: 32, worker of dataloader: 8, total cost: {tmp_recorder.get_total_duration_miliseconds()} ms, each batch extarct cost: {tmp_recorder2.get_avg_duration_miliseconds()} ms"
    )
    return feats_array
