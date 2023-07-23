# Introduction
This code is based off of the "microsoft GenerativeImage2Text" github project, which reproduces some of the results in the paper 
[GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100).

The code in the GIT repo does not provide support for finetuning/training the GIT model. Thus, the first step in this project was to adapt the GIT model to the Hugging Face Trainer framework.
This project also makes 3 modifications to the model:
1. An experiment with reducing the number of transformer layers and self-attention heads per layer in the text encoder component. (Found in the BertEncoderAsDecoder class)
2. An experiment with reducing the convolution stride size in the image encoder. (Found in the CLIP/model.VisualTransformer class)
3. An experiment to replace the text decoder (a modified version of BERT) with the GPT decoder. (Found in the TransformerDecoderTextualHead class)

# Installation

- Install the package
  ```shell
  pip install -r requirements.txt
  ```

# Training

- Training/captioning
  ```
  # Train model 1 - Ubuntu
  python ./train_hf.py \
    --output_dir ./git-model-baseline \
    --model_name_or_path ./ \
    --preprocessing_num_workers 12 \
    --data_dir /mnt/g/coco \
    --cache_dir /home/.../Datasets/coco/.cache \
	--train_file "instances_train2017.json" \
	--validation_file "instances_val2017.json" \
	--test_file "image_info_test2017.json" \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train  --do_eval \
    --per_device_train_batch_size="20" \
    --per_device_eval_batch_size="10" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir
  ```

# Citation
Please consider to cite the following reference if it helps.
```text
@article{wang2022git,
  title={GIT: A Generative Image-to-text Transformer for Vision and Language},
  author={Wang, Jianfeng and Yang, Zhengyuan and Hu, Xiaowei and Li, Linjie and Lin, Kevin and Gan, Zhe and Liu, Zicheng and Liu, Ce and Wang, Lijuan},
  journal={arXiv preprint arXiv:2205.14100},
  year={2022}
}
```

# Acknowledgement
Part of the code is based on
[transformers](https://github.com/huggingface/transformers),
[clip](https://github.com/openai/CLIP),
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark),
[oscar](https://github.com/microsoft/Oscar),
[virtex](https://github.com/kdexd/virtex).


