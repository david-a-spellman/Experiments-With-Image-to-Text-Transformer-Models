
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from generativeimage2text.common import Config
import json
import os.path as op
from generativeimage2text.common import qd_tqdm as tqdm
from generativeimage2text.common import json_dump
from generativeimage2text.common import pilimg_from_base64
from generativeimage2text.torch_common import recursive_to_device
from generativeimage2text.tsv_io import TSVFile, tsv_writer, tsv_reader
from generativeimage2text.common import write_to_file
from pprint import pformat
import logging
from transformers import BertTokenizer
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from azfuse import File
import numpy
import logging

import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from generativeimage2text.common import init_logging
from generativeimage2text.common import parse_general_args
from generativeimage2text.tsv_io import load_from_yaml_file
from generativeimage2text.torch_common import torch_load
from generativeimage2text.torch_common import load_state_dict
from generativeimage2text.torch_common import resize_2d_pos_embed
from generativeimage2text.layers.CLIP import clip
from generativeimage2text.layers.decoder import (TransformerDecoderTextualHead,
                             AutoRegressiveBeamSearch, GeneratorWithBeamSearch)
from generativeimage2text.layers.decoder import CaptioningModel
from generativeimage2text.process_image import load_image_by_pil
from generativeimage2text.data_layer.transform import RenameKey, SelectTransform
from generativeimage2text.data_layer.transform import ImageTransform2Dict
from generativeimage2text.data_layer.transform import get_inception_train_transform
from generativeimage2text.data_layer.builder import collate_fn
from generativeimage2text.model import get_git_model
import generativeimage2text.model
import generativeimage2text.model2
from pycocotools.coco import COCO
from wordfreq import word_frequency
from transformers import AutoTokenizer
import evaluate
from evaluate import load

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.29.0.dev0")

#require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default="./.cache/", metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters or not."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input testing data file (a jsonlines file)."},
    )
    max_seq_length: Optional[int] = field(
        default=40,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "json", "`validation_file` should be a json file."


dataset_name_mapping = {
    "image_caption_dataset.py": ("image_path", "caption"),
}

def get_data(image_file, prefix, target, tokenizer, image_transform):
    max_text_len = 40
    prefix_encoding = tokenizer(
        prefix, padding='do_not_pad',
        add_special_tokens=False,
        truncation=True, max_length=max_text_len)
    target_encoding = tokenizer(
        target, padding='do_not_pad',
        add_special_tokens=False,
        truncation=True, max_length=max_text_len)
    need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
    payload = prefix_encoding['input_ids'] + target_encoding['input_ids']
    if len(payload) > max_text_len:
        payload = payload[-(max_text_len - 2):]
        need_predict = need_predict[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload + [tokenizer.sep_token_id]
    need_predict = [0] + need_predict + [1]

    im = load_image_by_pil(image_file)

    data = {
        'caption_tokens': torch.tensor(input_ids),
        #'caption_lengths': len(input_ids),
        'need_predict': torch.tensor(need_predict),
        'image': im,
        # 'rect' field can be fed in 'caption', which tells the bounding box
        # -region of the image that is described by the caption. In this case,
        # we can optionally crop the region.
        'caption': {},
        # this iteration can be used for crop-size selection so that all GPUs
        # can process the image with the same input size
        'iteration': 0,
    }
    data = image_transform(data)

    return data

def get_image_transform(cfg):
    return get_multi_scale_image_transform(cfg, is_train=True)

def get_default_mean():
    return [0.485, 0.456, 0.406]

def get_default_std():
    return [0.229, 0.224, 0.225]

def get_transform_image_norm(cfg, default=None):
    if cfg.data_normalize == 'default':
        normalize = transforms.Normalize(
            mean=get_default_mean(), std=get_default_std())
    elif cfg.data_normalize == 'clip':
        # clip model
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    else:
        raise NotImplementedError(cfg.data_normalize)
    return normalize

def get_transform_vit_default(cfg, is_train):
    default_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = get_transform_image_norm(cfg, default_normalize)
    transform = get_inception_train_transform(
        bgr2rgb=True,
        crop_size=cfg.train_crop_size,
        normalize=normalize,
        small_scale=cfg.input_small_scale,
        no_color_jitter=cfg.no_color_jitter,
        no_flip=cfg.no_flip,
        no_aspect_dist=cfg.no_aspect_dist,
        resize_crop=cfg.resize_crop,
        max_size=cfg.train_max_size,
        interpolation=cfg.interpolation or Image.BILINEAR,
    )
    return transform

def get_transform_image(cfg, is_train):
    train_transform = cfg.train_transform
    if train_transform == 'vitp':
        transform = get_transform_vit_default(
            cfg, is_train=is_train)
    else:
        raise NotImplementedError(train_transform)
    return transform

class ImageTransform2Images(object):
    def __init__(self, sep_transform, first_joint=None):
        self.image_transform = sep_transform
        self.first_joint = first_joint

    def __call__(self, imgs):
        if self.first_joint is not None:
            imgs = self.first_joint(imgs)
        return [self.image_transform(im) for im in imgs]

    def __repr__(self):
        return 'ImageTransform2Images(image_transform={})'.format(
            self.image_transform,
        )

def get_transform_images(cfg, is_train):
    trans = get_transform_image(cfg, is_train)
    trans = ImageTransform2Images(trans)
    return trans

def trans_select_for_crop_size(
    data, train_crop_sizes,
    iteration_multi=0,
):
    return 0
    # if iteration_multi <= 0:
    #     if len(train_crop_sizes) == 1:
    #         idx = 0
    #     else:
    #         idx = data['iteration'] % len(train_crop_sizes)
    # elif data['iteration'] <= iteration_multi:
    #     idx = data['iteration'] % len(train_crop_sizes)
    # else:
    #     idx = -1
    # return idx

def get_multi_scale_image_transform(cfg, is_train, get_one=get_transform_image):
    def get_multi_res_transform(s):
        old = cfg.train_crop_size if is_train else cfg.test_crop_size
        all_t = []
        multi_res_factors = cfg.multi_res_factors or []
        for i, f in enumerate(multi_res_factors):
            if is_train:
                cfg.train_crop_size = s // f
            else:
                cfg.test_crop_size = s // f
            key = 'image_{}'.format(i)
            all_t.append(RenameKey({'image': key}, not_delete_origin=True))
            t = get_one(cfg, is_train)
            #t = ImageTransform2Dict(t, key=key)
            all_t.append(t)
        # get_one depends on train_crop_size
        if is_train:
            cfg.train_crop_size = s
        else:
            cfg.test_crop_size = s
        t = get_one(cfg, is_train)
        #t = ImageTransform2Dict(t)
        all_t.append(t)
        if is_train:
            cfg.train_crop_size = old
        else:
            cfg.test_crop_size = old
        return transforms.Compose(all_t)

    if is_train:
        if cfg.min_size_range32 is None:
            train_crop_sizes = [cfg.train_crop_size]
        else:
            train_crop_sizes = list(range(
                cfg.min_size_range32[0],
                cfg.min_size_range32[1] + cfg.patch_size - 1, cfg.patch_size,
            ))
    else:
        train_crop_sizes = [cfg.test_crop_size]

    crop_trans = []
    for s in train_crop_sizes:
        t = get_multi_res_transform(s)
        crop_trans.append(t)
    iteration_multi = 0
    image_transform = SelectTransform(
        crop_trans,
        lambda d: trans_select_for_crop_size(
            d, train_crop_sizes, iteration_multi))
    return image_transform

# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC, antialias=False),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x

def build_word_lut(tokenizer):
    words = list()
    with open("./generativeimage2text/spatial_words.txt", "r") as inFile:
        words = inFile.readlines()
    words = [w.strip("\n") for w in words]
    freqs = [word_frequency(w, 'en') for w in words]
    freqs = zip(words, freqs)
    freqs = [w_f for w_f in freqs]
    freqs.sort(key=lambda x: x[1], reverse=True)
    words_by_freq = [w_f[0] for w_f in freqs]
    # tokenize words
    word_tokens = tokenizer(words_by_freq, add_special_tokens=False, max_length=1)
    word_tokens = [w[0] for w in word_tokens["input_ids"]]
    print("Word tokens = " + str(word_tokens))
    return word_tokens


def collate_fn(examples):
    # pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    # attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    # return {
    #     "pixel_values": pixel_values,
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "return_loss": True,
    # }
    # output a dict of torch tensors 
    return {
        'caption_tokens': torch.tensor([example["input_ids"] for example in examples], dtype=torch.long),
        #'caption_lengths': len(input_ids),
        #'need_predict': torch.tensor([example["need_predict"] for example in examples], dtype=torch.int),
        'image': torch.stack([example['image'] for example in examples]),
        # 'rect' field can be fed in 'caption', which tells the bounding box
        # -region of the image that is described by the caption. In this case,
        # we can optionally crop the region.
        #'caption': {},
        # this iteration can be used for crop-size selection so that all GPUs
        # can process the image with the same input size
        #'iteration': 0,
    }

def evaluate_model(model,
                   tokenizer,
                   annotations_file,
                   image_dir,
                   max_text_len,
                   image_transform):
    cfg = {
        'crop_region_extend_in_datatransform': 4,
        'data_normalize': 'clip',
        'train_crop_size': 128,
        'input_small_scale': 0.8,
        'no_color_jitter': True,
        'no_flip': True,
        'no_aspect_dist': True,
        'interpolation': 'bicubic',
        'min_size_range32': [128, 224], # in pretraining, it is multi-scale from 160 to 224; while for fine-tuning, it is single scale
        'patch_size': 16,
        'train_transform': 'vitp',
    }
    cfg = Config(cfg, {})
    # image_transform = get_image_transform(cfg)
    bertscore = evaluate.load("bertscore")
    bleuscore = evaluate.load("bleu")

    num_examples = 1000
    print(f'Evaluate on {num_examples} images...')

    my_coco = COCO(annotations_file)
    ids = list(my_coco.anns.keys())
    print(f"Found {len(ids)} examples for eval.")
    curr_id_index = 0

    model.eval() # Put model in train mode
    model.cuda()  # Send model to GPU
    model.half()

    filter_words = build_word_lut(tokenizer)

    image_batch_size = 100
    num_batches = 1 #int(num_examples / image_batch_size)

    references = []
    predictions = []
    count = 1
    for b in range(0, num_batches):
        print(f"Batch {b}...")
        image_files = list()
        captions = list()

        while len(captions) < image_batch_size and curr_id_index < len(ids):
            id = ids[curr_id_index]
            curr_id_index += 1
            capts = my_coco.anns[id]["caption"]
            cap = ""
            if type(capts) == list:
                cap = capts[0]
            elif type(capts) == str:
                cap = capts
            else:
                continue
            
            cap_encoding = tokenizer(cap, padding='do_not_pad', add_special_tokens=False, truncation=True, max_length=max_text_len - 2)
            cap_encoding = [tokenizer.cls_token_id] + cap_encoding['input_ids'] + [tokenizer.sep_token_id]
 
            keep = False
            for t in cap_encoding:
                if t in filter_words:
                    keep = True
                    break
            if keep == True:
                captions.append(cap_encoding)
                references.append(cap)
                img_id = my_coco.anns[id]["image_id"]
                image_files.append(image_dir + my_coco.loadImgs(img_id)[0]["file_name"])

        prefixs = [''] * len(captions)

        print('Number of captions loaded: ' + str(len(captions)))

        all_data = []
        for image_file, prefix, cap_en in zip(image_files, prefixs, captions):
            #im = load_image_by_pil(image_file)
            im = read_image(image_file, mode=ImageReadMode.RGB)
            im = image_transform(im)
            im = torch.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
            data = { 'caption_tokens': torch.tensor(cap_en), 'image': im }
            all_data.append(data)

        #data = collate_fn(all_data)
        #logging.info(image_transform)
        all_data = recursive_to_device(all_data, 'cuda')

        for data in all_data:
            with torch.no_grad():
                result = model(caption_tokens=data['caption_tokens'], image=data['image'] )

            result_cap = tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
            predictions.append(result_cap)
            print(f"{count}:" + result_cap)
            count += 1

    with open("g:\\Projects\\results.txt", "w") as outfile:
        for ref, pred in zip(references, predictions):
            outfile.write(ref + ',' + pred + '\n')

    # Calc metrics
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")
    print("Bert Score: \n" + str(bert_results))

    # Calc glue
    bleu_results = bleuscore.compute(predictions=predictions, references=references)
    print("Bleu Score: \n" + str(bleu_results))


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.fp16 = True

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    #send_example_telemetry("run_clip", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventualy continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    # captions (unless you specify column names for this with the `image_column` and `caption_column` arguments).
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        

    # 5. Load model, tokenizer, and image processor
    # Initialize torchvision transforms and jit it for faster processing.
    mean=get_default_mean()
    image_std=get_default_std()
    image_transformations = Transform(96, mean, image_std)
    image_transformations = torch.jit.script(image_transformations)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    param = {}
    print(f"Creating model: {model_args.model_name_or_path}")
    if model_args.model_name_or_path == "baseline":
        model = generativeimage2text.model.get_git_model(tokenizer, param)
    elif model_args.model_name_or_path == "git-model-2":
        model = generativeimage2text.model2.get_git_model2(tokenizer, param) # removes 1/2 textual encoder layers and attention heads
    elif model_args.model_name_or_path == "git-model-3":
        model = generativeimage2text.model.get_git_model(tokenizer, param) # reduces stride of conv layer in image encoder
    elif model_args.model_name_or_path == "git-model-4":
        model = generativeimage2text.model.get_git_model(tokenizer, param, decoder_type="gpt")
    else:
        model = generativeimage2text.model.get_git_model(tokenizer, param)

       
    # Short circuit to special eval routine if only eval will be run
    if training_args.do_eval and not training_args.do_train:
        print("Running eval....")

        if model_args.model_name_or_path == "baseline":
            model.load_state_dict(torch.load("./git-model-baseline/pytorch_model.bin"))
        elif model_args.model_name_or_path == "model-2":
            model.load_state_dict(torch.load("./git-model-2/pytorch_model.bin"))
        elif model_args.model_name_or_path == "model-3":
            model.load_state_dict(torch.load("./git-model-3/pytorch_model.bin"))
        elif model_args.model_name_or_path == "model-4":
            model.load_state_dict(torch.load("./git-model-4/pytorch_model.bin"))
        else:
            model.load_state_dict(torch.load(model_args.model_name_or_path))

        # annotations_file = "/home/dspellman/Datasets/coco/image_info_test2017.json"
        annotations_file = "g:/Datasets/coco/captions_val2017.json"
        image_dir = "g:/Datasets/coco/val2017/"
        evaluate_model(model,
                   tokenizer,
                   annotations_file,
                   image_dir,
                   40,
                   image_transformations)
        quit()

    COCO_DIR = data_args.data_dir
    dataset = load_dataset("ydshieh/coco_dataset_script", "2017",
                           cache_dir=model_args.cache_dir,
                           data_dir=COCO_DIR)

    word_lut = build_word_lut(tokenizer)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)

    if model_args.freeze_text_model:
        _freeze_params(model.text_model)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(data_args.dataset_name, None)
    if data_args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = data_args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{data_args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # 7. Preprocessing the datasets.
    # Load image_processor, in this script we only use this to get the mean and std for normalization.
    # image_processor = AutoImageProcessor.from_pretrained(
    #     model_args.image_processor_name or model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )

    # Initialize torchvision transforms and jit it for faster processing.
    mean=get_default_mean()
    image_std=get_default_std()
    image_transformations = Transform(128, mean, image_std)
    # cfg = {
    #     'crop_region_extend_in_datatransform': 4,
    #     'data_normalize': 'clip',
    #     'train_crop_size': 224,
    #     'input_small_scale': 0.8,
    #     'no_color_jitter': True,
    #     'no_flip': True,
    #     'no_aspect_dist': True,
    #     'interpolation': 'bicubic',
    #     'min_size_range32': [160, 224], # in pretraining, it is multi-scale from 160 to 224; while for fine-tuning, it is single scale
    #     'patch_size': 16,
    #     'train_transform': 'vitp',
    # }
    # cfg = Config(cfg, {})
    # image_transformations = get_image_transform(cfg)
    image_transformations = torch.jit.script(image_transformations)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(captions, add_special_tokens=False, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        input_ids = [[tokenizer.cls_token_id] + list_ids + [tokenizer.sep_token_id] for list_ids in text_inputs.input_ids]
        examples["input_ids"] = input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def transform_images(examples):
        #images = [load_image_by_pil(image_file) for image_file in examples[image_column]]
        images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
        examples["image"] = [image_transformations(image) for image in images]
        return examples

    def filter_bad_images(examples):
        """remove problematic images"""
        valid_images = [False] * len(examples[image_column])
        #for image_file in examples[image_column]:
        for i in range(len(valid_images)):
            image_file = examples[image_column][i]
            caption = examples["input_ids"][i]
            try:
                Image.open(image_file)
                for token in word_lut:
                    if token in caption:
                        valid_images[i] = True
                        continue
            except Exception:
                continue
        return valid_images

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        train_dataset = train_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        train_dataset = train_dataset.filter(
            filter_bad_images, batched=True, num_proc=data_args.preprocessing_num_workers
        )
        # Transform images on the fly as doing it on the whole dataset takes too much time.
        train_dataset.set_transform(transform_images)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        eval_dataset = eval_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        eval_dataset = eval_dataset.filter(
            filter_bad_images, batched=True, num_proc=data_args.preprocessing_num_workers
        )
        # Transform images on the fly as doing it on the whole dataset takes too much time.
        eval_dataset.set_transform(transform_images)

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = dataset["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(test_dataset), data_args.max_eval_samples)
            test_dataset = test_dataset.select(range(max_eval_samples))
        test_dataset = test_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )
        test_dataset = test_dataset.filter(
            filter_bad_images, batched=True, num_proc=data_args.preprocessing_num_workers
        )
        # Transform images on the fly as doing it on the whole dataset takes too much time.
        test_dataset.set_transform(transform_images)

    def compute_metrics(eval_preds):
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 8. Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. Write Training Stats and push to hub.
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "contrastive-image-text-modeling"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()




