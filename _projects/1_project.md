---
layout: page
title: Efficient Self-supervised Vision Transformers
description: Learning visual representation from unlabelled images
importance: 1
category: work
---

Self-supervised learning (SSL) with Transformers has become a de facto standard of model choice in natural language processing (NLP). The dominant approaches such as GPT and BERT are pre-training on a large text corpus and then fine-tuning to various smaller task-specific datasets, showing superior performance. In computer vision (CV), however, self-supervised visual representation learning is still dominated by convolutional neural networks (CNNs). Sharing a similar goal/spirit with NLP, SSL in CV aims to learn general-purpose image features from raw pixels without relying on manual supervisions. Prior SoTA methods on linear probe evaluation of ImageNet-1K is achieved, by exhaustively consuming computation resource on full self-attention operators with long sequences of split image patches. Aiming to improve the efficiency of Transformer-based SSL in CV paves the way to achieve Green AI.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/1.jpg title: "example image" class: "img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/3.jpg title: "example image" class: "img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/5.jpg title: "example image" class: "img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Self-supervised learning (SSL) in computer vision aims to learn general-purpose image features from raw pixels without relying on manual supervisions,
and the learned networks serve as the backbone of various downstream tasks such as classification,
detection and segmentation
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% responsive_image path: assets/img/esvit_sota.png title: "example image" class: "img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Efficiency vs accuracy comparison under the linear classification protocol on ImageNet. Left: Throughput of all SoTA SSL vision systems, circle sizes indicates model parameter counts; Right: performance over varied parameter counts for models with moderate (throughout/#parameters) ratio. Please refer Section 4.1 for details.
</div>


PyTorch implementation for EsViT, built with two techniques:

- A multi-stage Transformer architecture. Three multi-stage Transformer variants are implemented under the folder `models`.
- A non-contrastive region-level matching pre-train task. The region-level matching task is implemented in function RegionMathcingLoss(nn.Module) in main_esvit.py. Please use `--use_dense_prediction True`, otherwise only the view-level task is used.


The code of the non-contrastive region-level matching pre-train task is simple. Just add `RegionMathcingLoss` into the view-level pre-train loss. Here's the code:

{% raw %}
```python
class RegionMathcingLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_grid", torch.zeros(1, out_dim))


        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, targets_mixup):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """


        s_cls_out, s_region_out, s_fea, s_npatch = student_output
        t_cls_out, t_region_out, t_fea, t_npatch = teacher_output

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        t_cls = F.softmax((t_cls_out - self.center) / temp, dim=-1)
        t_cls = t_cls.detach().chunk(2)

        t_region = F.softmax((t_region_out - self.center_grid) / temp, dim=-1)
        t_region = t_region.detach().chunk(2)
        t_fea = t_fea.chunk(2)

        
        N = t_npatch[0] # num of patches in the first view
        B = t_region[0].shape[0]//N # batch size, 

        # student sharpening
        s_cls = s_cls_out / self.student_temp
        s_cls = s_cls.chunk(self.ncrops)

        s_region = s_region_out / self.student_temp
        s_split_size = [s_npatch[0]] * 2 + [s_npatch[1]] * (self.ncrops -2) 
        
        s_split_size_bs = [i * B for i in s_split_size]
        
        s_region = torch.split(s_region, s_split_size_bs, dim=0)
        s_fea = torch.split(s_fea, s_split_size_bs, dim=0)



        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(t_cls):
            for v in range(len(s_cls)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                
                # view level prediction loss
                loss = 0.5 * torch.sum(-q * F.log_softmax(s_cls[v], dim=-1), dim=-1)

                # region level prediction loss
                s_region_cur, s_fea_cur = s_region[v].view(B, s_split_size[v], -1), s_fea[v].view(B, s_split_size[v], -1)  # B x T_s x K, B x T_s x P
                t_region_cur, t_fea_cur = t_region[iq].view(B, N, -1), t_fea[iq].view(B, N, -1)  # B x T_t x K, B x T_t x P, 

                # similarity matrix between two sets of region features
                region_sim_matrix = torch.matmul(F.normalize(s_fea_cur, p=2, dim=-1) , F.normalize(t_fea_cur, p=2, dim=-1) .permute(0, 2, 1)) # B x T_s x T_t
                region_sim_ind = region_sim_matrix.max(dim=2)[1] # B x T_s; collect the argmax index in teacher for a given student feature
                
                t_indexed_region = torch.gather( t_region_cur, 1, region_sim_ind.unsqueeze(2).expand(-1, -1, t_region_cur.size(2)) ) # B x T_s x K (index matrix: B, T_s, 1)

                loss_grid = torch.sum(- t_indexed_region * F.log_softmax(s_region_cur, dim=-1), dim=[-1]).mean(-1)   # B x T_s x K --> B 

                loss += 0.5 * loss_grid

                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        self.update_center(t_cls_out, t_region_out)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output, teacher_grid_output):
        """
        Update center used for teacher output.
        """

        # view level center update
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # region level center update
        batch_grid_center = torch.sum(teacher_grid_output, dim=0, keepdim=True)
        dist.all_reduce(batch_grid_center)
        batch_grid_center = batch_grid_center / (len(teacher_grid_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        self.center_grid = self.center_grid * self.center_momentum + batch_grid_center * (1 - self.center_momentum)
```
{% endraw %}
