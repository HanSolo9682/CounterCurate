# CounterCurate


This is the implementation of CounterCurate, the data curation pipeline of both physical and semantic counterfactual image-caption pairs.

\[[Paper](https://arxiv.org/abs/2402.13254)\]\[[Project Page](https://countercurate.github.io/)\]\[[HuggingFace](https://huggingface.co/datasets/HanSolo9682/Flickr30k-Counterfactuals)\]

In our paper, we used Flickr30k-Images and the Visual Genome dataset. If you want to fully test out CounterCurate, you must download these two datasets from their authors into our `datasets` folder.

\[[Flickr30k-Images](https://shannon.cs.illinois.edu/DenotationGraph/)\]\[[VisualGenome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)\]

We further employ the following GitHub repositories, please clone them into the `datasets` folder for demo purposes.

\[[Flickr30k-Entities](https://github.com/BryanPlummer/flickr30k_entities)\]\[[PointQA](https://github.com/princetonvisualai/pointingqa/tree/main)\]\[[SugarCrepe](https://github.com/RAIVNLab/sugar-crepe)\]

For further training and testing, please use the following:

\[[OpenCLIP](https://github.com/mlfoundations/open_clip)\]\[[LLaVA](https://github.com/haotian-liu/LLaVA)\]\[[GLIGEN](https://github.com/gligen/GLIGEN)\]

For all the code we provide, you can find a few lines of editable parameters right after the import statements. Please change them accordingly to adjust to your environment.

## Datasets
Please clone our HuggingFace repository, which contains our generated training data (`train_data.tar.gz`), sample data that can be used to train CLIP (`clip_sample_data/*.csv`)/LLaVA (`llava_sample_data/*.json`), and our GPT-4V generated prompts which aided us in the data curation process (`gpt4v_*.json`).

Note that the data in HuggingFace does not contain any original images from Flickr30k-Images. You must first run
```bash
python datasets/prepare_training_data.py
```
to copy all the original images into their respective positions in the `train_data` folder. After this, you can technically begin training/testing. All other files in the `datasets` folder are used to curate the generated data.
- `generate_positions.py`: generates negative captions for Flickr30k-Positions;
- `generate_counting.py`: generates negative captions for Flickr30k-Counting;
- `prepare_original_data.py`: generates a json file containing each image and its 5 captions from Flickr30k-Images;
- `mark_imgs.py`: marks the original Flickr30k-Images with bounding boxes to be fed into GPT4-V;
- `dalle_gen.py`: generates negative images for Flickr30k-Attributes;
- `gpt4v_prompt_gen.py`: generates negative captions for Flickr30k-Attributes;
- `gpt4v_dalle.py`: bridges between GPT-4V and DALLE-3 by filtering and converting GPT's response to be ready to be fed into DALLE;
- `gpt4v_updown_gen.py`: generates expanded prompts for GLIGEN;
- `gligen_inference_count.py`: You can use this file to reproduce our data curation for Flickr30k-Counting;
- `gligen_inference_up_down.py`: You can use this file to reproduce our data curation for Flickr30k-Position's up-down subset.

## Benchmarking
- `prep_count_benchmark.py`: creates our simple benchmark testing model's counting capabilities, saved to a file `vg_dict.json`;
- `CLIP_test_position.py`: uses Open-CLIP to test a CLIP model's physical compositional understanding on Flickr30k-Positions;
- `CLIP_test_counting.py`: uses Open-CLIP to test a CLIP model's counting capabilities with `vg_dict.json`;
- `llava_evaluate_counting.py`: evaluates a LLaVA model's counting capabilities with `vg_dict.json`;
- `llava_evaluate_position.py`: evaluates a LLaVA model's physical compositional understanding on Flickr30k-Positions;
- `llava_evaluagte_sugarcrepe.py`: evaluates a LLaVA model's semantical compositional understanding on SugarCrepe.

## Training
- `prep_clip_attr.py`: generates a csv file that OpenCLIP accepts as input training data for Flickr30k-Attributes;
- `prep_clip_pos_count.py`: generates a csv file that OpenCLIP accepts as input training data for Flickr30k-Positions/Flickr30k-Counting;
- `prep_training_data_llava.py`: generates a json file that LLaVA accepts as input training data.

## Grouping
In the paper, we used grouping to force CLIP models to learn the positive and negative image/caption pairs of the same index within the same batch to improve training efficiency and accuracy. Here, we provide two versions of grouping we used to train our CLIP models. The folder `train/grouping` is the general grouping strategy for positive/negative image-caption pairs, or when there is only one positive and one negative per item; the folder `train/grouping_attr` is specifically taylored for training our model on Flickr30k-Attributes, where each positive (original) image-caption pair has 2-3 negative counterfactuals (noun, adjective, reverse). Both folders contain a `data.py` and a `train.py`. To reproduce, please replace the files under your OpenClip repository `open_clip/src/training/.../py` accordingly. 

## Sample Data
It is to note that the sample data files have specific path formats (such as `../../train_data`). This is because of the testing environment we evaluated CounterCurate on. Please feel free alter the file paths.

## Citation

If you find CounterCurate useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{zhang2024countercurate,
  title={CounterCurate: Enhancing Physical and Semantic Visio-Linguistic Compositional Reasoning via Counterfactual Examples},
  author={Zhang, Jianrui and Cai, Mu and Xie, Tengyang and Lee, Yong Jae},
  journal={arXiv preprint arXiv:2402.13254},
  year={2024}
}
```



## Acknowledgement

- [CLIP](https://openai.com/research/clip): the contrastive multimodal model we utilized. 

- [LLaVa](https://llava-vl.github.io/):  the generative multimodal model we utilized. 

