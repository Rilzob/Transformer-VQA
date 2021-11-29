import torch
import torch.nn as nn
import torch.nn.functional as F
from language_model import WordEmbedding, QuestionEmbedding
from net_utils import FC, MLP, LayerNorm
import numpy as np
import math
from classifier import SimpleClassifier
import random


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, opt):
        super(MHAtt, self).__init__()

        self.opt = opt

        self.linear_v = nn.Linear(opt.HIDDEN_SIZE, opt.HIDDEN_SIZE)
        self.linear_k = nn.Linear(opt.HIDDEN_SIZE, opt.HIDDEN_SIZE)
        self.linear_q = nn.Linear(opt.HIDDEN_SIZE, opt.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(opt.HIDDEN_SIZE, opt.HIDDEN_SIZE)

        self.dropout = nn.Dropout(opt.dropL)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt.MULTI_HEAD,
            self.opt.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt.MULTI_HEAD,
            self.opt.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt.MULTI_HEAD,
            self.opt.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# ---------------------------
# -- Feed Forward Network ---
# ---------------------------

class FFN(nn.Module):
    def __init__(self, opt):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=opt.HIDDEN_SIZE,
            mid_size=opt.FF_SIZE,
            out_size=opt.HIDDEN_SIZE,
            dropout_r=opt.dropL,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ------- Encoder --------
# ------------------------

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.mhatt = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt.dropL)
        self.norm1 = LayerNorm(opt.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(opt.dropL)
        self.norm2 = LayerNorm(opt.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ----------- Decoder -----------
# -------------------------------

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()

        self.mhatt1 = MHAtt(opt)
        self.mhatt2 = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt.dropL)
        self.norm1 = LayerNorm(opt.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(opt.dropL)
        self.norm2 = LayerNorm(opt.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(opt.dropL)
        self.norm3 = LayerNorm(opt.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        self.enc_list = nn.ModuleList([Encoder(opt) for _ in range(opt.LAYER)])
        self.dec_list = nn.ModuleList([Decoder(opt) for _ in range(opt.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, opt):
        super(AttFlat, self).__init__()
        self.opt = opt

        self.mlp = MLP(
            in_size=opt.HIDDEN_SIZE,
            mid_size=opt.FLAT_MLP_SIZE,
            out_size=opt.FLAT_GLIMPSES,
            dropout_r=opt.dropL,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            opt.HIDDEN_SIZE * opt.FLAT_GLIMPSES,
            opt.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.opt.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        num_hid = opt.num_hid
        activation = opt.activation
        # dropout = opt.dropout
        dropL = opt.dropL
        norm = opt.norm
        dropC = opt.dropC
        self.opt = opt

        self.embedding = nn.Embedding(
            num_embeddings=opt.ntokens+1,
            embedding_dim=opt.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        weight_init = torch.from_numpy(np.load(opt.dataroot + 'glove6b_init_300d.npy'))
        self.embedding.weight.data[:weight_init.shape[0]] = weight_init

        self.lstm = nn.LSTM(
            input_size=opt.WORD_EMBED_SIZE,
            hidden_size=opt.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            opt.IMG_FEAT_SIZE,
            opt.HIDDEN_SIZE
        )

        self.backbone = Net(opt)
        self.attflat_img = AttFlat(opt)
        self.attflat_lang = AttFlat(opt)

        self.proj_norm = LayerNorm(opt.FLAT_OUT_SIZE)
        self.proj = nn.Linear(opt.FLAT_OUT_SIZE, opt.ans_dim)


    def forward(self, question, img_feat, self_sup=True):

        # Pre-process Language Feature
        lang_feat = self.embedding(question)
        # print("lang_feat_1.shape:", lang_feat.shape)
        lang_feat, _ = self.lstm(lang_feat)
        # print("lang_feat_2.shape:", lang_feat.shape)

        # Pre-process Image Feature
        # img_feat = self.img_feat_linear(img_feat)
        # print("img_feat.shape:", img_feat.shape)

        # Make mask
        # lang_feat_mask = self.make_mask(question.unsqueeze(2))
        # print("lang_feat_mask.shape:", lang_feat_mask.shape)
        # img_feat_mask = self.make_mask(img_feat)
        # print("img_feat_mask.shape:", img_feat_mask.shape)


        logits_pos, att_gv_pos = self.compute_predict(lang_feat, img_feat, question)

        batch_size = question.size(0)

        if self_sup:
            # construct an irrelevant Q-I pair for each instance
            index = random.sample(range(0, batch_size), batch_size)
            img_feat_neg = img_feat[index]
            logits_neg, att_gv_neg = self.compute_predict(lang_feat, img_feat_neg, question)
            return logits_pos, logits_neg, att_gv_pos, att_gv_neg
        else:
            return logits_pos, att_gv_pos


    def compute_predict(self, lang_feat, img_feat, ques_ix):
        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat_result = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat_result)
        proj_feat = self.proj(proj_feat)

        return proj_feat, proj_feat_result

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 16800).unsqueeze(1).unsqueeze(2)