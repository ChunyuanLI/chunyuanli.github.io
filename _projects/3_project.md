---
layout: page
title: Vision and Language Pre-training
description: Enrich cross-modal representations by connecting image understanding with rich semantics from language
img: assets/img/7.jpg
redirect: https://www.microsoft.com/en-us/research/blog/objects-are-the-secret-key-to-revealing-the-world-between-vision-and-language/
importance: 3
category: work
---

Humans perceive the world through many channels, such as images viewed by the eyes or voices heard by the ears. Though any individual channel might be incomplete or noisy, humans can naturally align and fuse the information collected from multiple channels to grasp the key concepts needed for a better understanding of the world. One of the core aspirations in artificial intelligence is to develop algorithms that endow computers with an ability to effectively learn from multi-modality (or multi-channel) data, similar to sights and sounds attained from vision and language that help humans make sense of the world around us. For example, computers could mimic this ability by searching the most similar images for a text query (or vice versa) and describing the content of an image using natural language.

Recently, vision-and-language pre-training (VLP) has shown great progress toward addressing this problem. The most representative approach is to train large Transformer-based models on massive image-text pair data in a self-supervised manner, such as predicting the masked elements based on their context. The cross-modal representations of the pre-training models can be fine-tuned to adapt to various downstream vision-and-language tasks. However, existing VLP methods simply concatenate image region features and text features as input to the model for pre-training and use self-attention to learn image-text semantic alignments in a brute-force yet implicit manner, leaving the model to figure out the cross-modal alignment from scratch.

In this blog post, we introduce Oscar (Object-Semantics Aligned Pre-training) to highlight our observation that objects can be naturally used as anchor points to ease the learning of semantic alignments between images and texts. This discovery leads to a novel VLP framework that creates new state-of-the-art performance on six well-established vision-and-language tasks.



