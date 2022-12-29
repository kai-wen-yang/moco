# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pdb

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, num_target_classes=1000):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, num_target_classes=num_target_classes)
        self.encoder_k = base_encoder(num_classes=dim, num_target_classes=num_target_classes)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

    def init_teacher(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, im_q, im_k=None, is_eval=False, is_target=False, centroids=None, index=None,  temp=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        if is_eval:
            # compute query features
            q = self.encoder_q(im_q)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)

            return q

        if is_target:
            logits = self.encoder_q(im_q, is_target=True)

            return logits
        else:

            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()
                k = self.encoder_k(im_q)
            q = self.encoder_q(im_k)
            k = nn.functional.normalize(k, dim=1)
            q = nn.functional.normalize(q, dim=1)
            logits_q = torch.mm(q, centroids.t())/temp
            logits_k = torch.mm(k, centroids.t())/temp

            #print(logits_k.max(dim=1))
            pseudo_label = torch.softmax(logits_k.detach(), dim=-1)
            max_probs, labels_proto = torch.max(pseudo_label, dim=-1)
            logits_q=logits_q#/density
            sim, _ = logits_k.max(dim=1)

            return logits_q, sim, labels_proto, max_probs



    # def forward(self, im_q, im_k):
    #     """
    #     Input:
    #         im_q: a batch of query images
    #         im_k: a batch of key images
    #     Output:
    #         logits, targets
    #     """
    #
    #     # compute query features
    #     q = self.encoder_q(im_q)  # queries: NxC
    #     q = nn.functional.normalize(q, dim=1)
    #
    #     # compute key features
    #     with torch.no_grad():  # no gradient to keys
    #         self._momentum_update_key_encoder()  # update the key encoder
    #
    #         # shuffle for making use of BN
    #         im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
    #
    #         k = self.encoder_k(im_k)  # keys: NxC
    #         k = nn.functional.normalize(k, dim=1)
    #
    #         # undo shuffle
    #         k = self._batch_unshuffle_ddp(k, idx_unshuffle)
    #
    #     # compute logits
    #     # Einstein sum is more intuitive
    #     # positive logits: Nx1
    #     l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    #     # negative logits: NxK
    #     l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
    #
    #     # logits: Nx(1+K)
    #     logits = torch.cat([l_pos, l_neg], dim=1)
    #
    #     # apply temperature
    #     logits /= self.T
    #
    #     # labels: positive key indicators
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    #
    #     # dequeue and enqueue
    #     self._dequeue_and_enqueue(k)
    #
    #     return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
