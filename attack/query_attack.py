from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_attacker import Attack

from query.surrogate import *
from query.victim import *
import prettytable as pt



# Square ###############################################################
class Square(Attack):
    r"""
    Square Attack in the paper 'Square Attack: a query-efficient black-box adversarial attack via random search'
    [https://arxiv.org/abs/1912.00049]
    [https://github.com/fra31/auto-attack]
    Distance Measure : Linf, L2
    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        n_queries (int): max number of queries (each restart). (Default: 5000)
        n_restarts (int): number of random restarts. (Default: 1)
        p_init (float): parameter to control size of squares. (Default: 0.8)
        loss (str): loss function optimized ['margin', 'ce'] (Default: 'margin')
        resc_schedule (bool): adapt schedule of p to n_queries (Default: True)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)
        targeted (bool): targeted. (Default: False)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = attack.Square(model, norm='Linf', eps=8/255, n_queries=5000, n_restarts=1, p_init=.8, seed=0, verbose=False, targeted=False, loss='margin', resc_schedule=True)
        >>> adv_images = attack(images, labels)
    """

    def __init__(
        self,
        model,
        norm="Linf",
        eps=8 / 255,
        n_queries=5000,
        n_restarts=1,
        p_init=0.8,
        loss="margin",
        resc_schedule=True,
        seed=0,
        verbose=False,
        targeted=False,
    ):
        super().__init__("Square", model)
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.supported_mode = ["default", "targeted"]
        self.targeted = targeted

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.perturb(images, labels)
        return adv_images

    def margin_and_loss(self, x, y):
        """
        :param y:        correct labels if untargeted else target labels
        """
        logits = self.get_logits(x)
        xent = F.cross_entropy(logits, y, reduction="none")
        u = torch.arange(x.shape[0])
        y_corr = logits[u, y].clone()
        logits[u, y] = -float("inf")
        y_others = logits.max(dim=-1)[0]

        if not self.targeted:
            if self.loss == "ce":
                return y_corr - y_others, -1.0 * xent
            elif self.loss == "margin":
                return y_corr - y_others, y_corr - y_others
        else:
            if self.loss == "ce":
                return y_others - y_corr, xent
            elif self.loss == "margin":
                return y_others - y_corr, y_others - y_corr

    def init_hyperparam(self, x):
        assert self.norm in ["Linf", "L2"]
        assert not self.eps is None
        assert self.loss in ["ce", "margin"]
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def normalize_delta(self, x):
        if self.norm == "Linf":
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)
        elif self.norm == "L2":
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def lp_norm(self, x):
        if self.norm == "L2":
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def eta_rectangles(self, x, y):
        delta = torch.zeros([x, y]).to(self.device)
        x_c, y_c = x // 2 + 1, y // 2 + 1
        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[
                max(counter2[0], 0) : min(counter2[0] + (2 * counter + 1), x),
                max(0, counter2[1]) : min(counter2[1] + (2 * counter + 1), y),
            ] += (
                1.0 / (torch.Tensor([counter + 1]).view(1, 1).to(self.device) ** 2)
            )  # nopep8
            counter2[0] -= 1
            counter2[1] -= 1
        delta /= (delta ** 2).sum(dim=(0, 1), keepdim=True).sqrt()
        return delta

    def eta(self, s):
        delta = torch.zeros([s, s]).to(self.device)
        delta[: s // 2] = self.eta_rectangles(s // 2, s)
        delta[s // 2 :] = -1.0 * self.eta_rectangles(s - s // 2, s)
        delta /= (delta ** 2).sum(dim=(0, 1), keepdim=True).sqrt()
        if torch.rand([1]) > 0.5:
            delta = delta.permute([1, 0])
        return delta

    def p_selection(self, it):
        """ schedule to decrease the parameter p """
        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)
        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it:
            p = self.p_init / 512
        else:
            p = self.p_init
        return p

    def attack_single_run(self, x, y):
        with torch.no_grad():
            adv = x.clone()
            c, h, w = x.shape[1:]
            n_features = c * h * w
            n_ex_total = x.shape[0]

            if self.norm == "Linf":
                x_best = torch.clamp(
                    x + self.eps * self.random_choice([x.shape[0], c, 1, w]), 0.0, 1.0
                )
                margin_min, loss_min = self.margin_and_loss(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)
                s_init = int(math.sqrt(self.p_init * n_features / c))

                for i_iter in range(self.n_queries):
                    idx_to_fool = (margin_min > 0.0).nonzero().flatten()
                    if len(idx_to_fool) == 0:
                        break
                    x_curr = self.check_shape(x[idx_to_fool])
                    x_best_curr = self.check_shape(x_best[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr = y_curr.unsqueeze(0)
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]

                    p = self.p_selection(i_iter)
                    s = max(int(round(math.sqrt(p * n_features / c))), 1)
                    vh = self.random_int(0, h - s)
                    vw = self.random_int(0, w - s)
                    new_deltas = torch.zeros([c, h, w]).to(self.device)
                    new_deltas[:, vh : vh + s, vw : vw + s] = (
                        2.0 * self.eps * self.random_choice([c, 1, 1])
                    )

                    x_new = x_best_curr + new_deltas
                    x_new = torch.min(
                        torch.max(x_new, x_curr - self.eps), x_curr + self.eps
                    )
                    x_new = torch.clamp(x_new, 0.0, 1.0)
                    x_new = self.check_shape(x_new)

                    margin, loss = self.margin_and_loss(x_new, y_curr)
                    # update loss if new loss is better
                    idx_improved = (loss < loss_min_curr).float()
                    loss_min[idx_to_fool] = (
                        idx_improved * loss + (1.0 - idx_improved) * loss_min_curr
                    )

                    # update margin and x_best if new loss is better
                    # or misclassification
                    idx_miscl = (margin <= 0.0).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)

                    margin_min[idx_to_fool] = (
                        idx_improved * margin + (1.0 - idx_improved) * margin_min_curr
                    )
                    idx_improved = idx_improved.reshape([-1, *[1] * len(x.shape[:-1])])
                    x_best[idx_to_fool] = (
                        idx_improved * x_new + (1.0 - idx_improved) * x_best_curr
                    )
                    n_queries[idx_to_fool] += 1.0

                    ind_succ = (margin_min <= 0.0).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        print(
                            "{}".format(i_iter + 1),
                            "- success rate={}/{} ({:.2%})".format(
                                ind_succ.numel(),
                                n_ex_total,
                                float(ind_succ.numel()) / n_ex_total,
                            ),
                            "- avg # queries={:.1f}".format(
                                n_queries[ind_succ].mean().item()
                            ),
                            "- med # queries={:.1f}".format(
                                n_queries[ind_succ].median().item()
                            ),
                            "- loss={:.3f}".format(loss_min.mean()),
                        )

                    if ind_succ.numel() == n_ex_total:
                        break

            elif self.norm == "L2":
                delta_init = torch.zeros_like(x)
                s = h // 5
                sp_init = (h - s * 5) // 2
                vh = sp_init + 0
                for _ in range(h // s):
                    vw = sp_init + 0
                    for _ in range(w // s):
                        delta_init[:, :, vh : vh + s, vw : vw + s] += self.eta(s).view(
                            1, 1, s, s
                        ) * self.random_choice([x.shape[0], c, 1, 1])
                        vw += s
                    vh += s

                x_best = torch.clamp(
                    x + self.normalize_delta(delta_init) * self.eps, 0.0, 1.0
                )
                margin_min, loss_min = self.margin_and_loss(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)
                s_init = int(math.sqrt(self.p_init * n_features / c))

                for i_iter in range(self.n_queries):
                    idx_to_fool = (margin_min > 0.0).nonzero().flatten()
                    if len(idx_to_fool) == 0:
                        break

                    x_curr = self.check_shape(x[idx_to_fool])
                    x_best_curr = self.check_shape(x_best[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr = y_curr.unsqueeze(0)
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]

                    delta_curr = x_best_curr - x_curr
                    p = self.p_selection(i_iter)
                    s = max(int(round(math.sqrt(p * n_features / c))), 3)
                    if s % 2 == 0:
                        s += 1

                    vh = self.random_int(0, h - s)
                    vw = self.random_int(0, w - s)
                    new_deltas_mask = torch.zeros_like(x_curr)
                    new_deltas_mask[:, :, vh : vh + s, vw : vw + s] = 1.0
                    norms_window_1 = (
                        (delta_curr[:, :, vh : vh + s, vw : vw + s] ** 2)
                        .sum(dim=(-2, -1), keepdim=True)
                        .sqrt()
                    )

                    vh2 = self.random_int(0, h - s)
                    vw2 = self.random_int(0, w - s)
                    new_deltas_mask_2 = torch.zeros_like(x_curr)
                    new_deltas_mask_2[:, :, vh2 : vh2 + s, vw2 : vw2 + s] = 1.0

                    norms_image = self.lp_norm(x_best_curr - x_curr)
                    mask_image = torch.max(new_deltas_mask, new_deltas_mask_2)
                    norms_windows = self.lp_norm(delta_curr * mask_image)

                    new_deltas = torch.ones([x_curr.shape[0], c, s, s]).to(self.device)
                    new_deltas *= self.eta(s).view(1, 1, s, s) * self.random_choice(
                        [x_curr.shape[0], c, 1, 1]
                    )
                    old_deltas = delta_curr[:, :, vh : vh + s, vw : vw + s] / (
                        1e-12 + norms_window_1
                    )
                    new_deltas += old_deltas
                    new_deltas = (
                        new_deltas
                        / (
                            1e-12
                            + (new_deltas ** 2).sum(dim=(-2, -1), keepdim=True).sqrt()
                        )
                        * (
                            torch.max(
                                (self.eps * torch.ones_like(new_deltas)) ** 2
                                - norms_image ** 2,
                                torch.zeros_like(new_deltas),
                            )
                            / c
                            + norms_windows ** 2
                        ).sqrt()
                    )
                    delta_curr[:, :, vh2 : vh2 + s, vw2 : vw2 + s] = 0.0
                    delta_curr[:, :, vh : vh + s, vw : vw + s] = new_deltas + 0

                    x_new = torch.clamp(
                        x_curr + self.normalize_delta(delta_curr) * self.eps, 0.0, 1.0
                    )
                    x_new = self.check_shape(x_new)
                    norms_image = self.lp_norm(x_new - x_curr)

                    margin, loss = self.margin_and_loss(x_new, y_curr)
                    # update loss if new loss is better
                    idx_improved = (loss < loss_min_curr).float()
                    loss_min[idx_to_fool] = (
                        idx_improved * loss + (1.0 - idx_improved) * loss_min_curr
                    )

                    # update margin and x_best if new loss is better
                    # or misclassification
                    idx_miscl = (margin <= 0.0).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)

                    margin_min[idx_to_fool] = (
                        idx_improved * margin + (1.0 - idx_improved) * margin_min_curr
                    )
                    idx_improved = idx_improved.reshape([-1, *[1] * len(x.shape[:-1])])
                    x_best[idx_to_fool] = (
                        idx_improved * x_new + (1.0 - idx_improved) * x_best_curr
                    )
                    n_queries[idx_to_fool] += 1.0

                    ind_succ = (margin_min <= 0.0).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        print(
                            "{}".format(i_iter + 1),
                            "- success rate={}/{} ({:.2%})".format(
                                ind_succ.numel(),
                                n_ex_total,
                                float(ind_succ.numel()) / n_ex_total,
                            ),
                            "- avg # queries={:.1f}".format(
                                n_queries[ind_succ].mean().item()
                            ),
                            "- med # queries={:.1f}".format(
                                n_queries[ind_succ].median().item()
                            ),
                            "- loss={:.3f}".format(loss_min.mean()),
                        )

                    assert (x_new != x_new).sum() == 0
                    assert (x_best != x_best).sum() == 0

                    if ind_succ.numel() == n_ex_total:
                        break

        return n_queries, x_best

    def perturb(self, x, y=None):
        """
        :param x:           clean images
        :param y:           untargeted attack -> clean labels,
                            if None we use the predicted labels
                            targeted attack -> target labels, if None random classes,
                            different from the predicted ones, are sampled
        """
        self.init_hyperparam(x)
        adv = x.clone()
        if y is None:
            if not self.targeted:
                with torch.no_grad():
                    output = self.get_logits(x)
                    y_pred = output.max(1)[1]
                    y = y_pred.detach().clone().long().to(self.device)
            else:
                with torch.no_grad():
                    y = self.get_target_label(x, None)
        else:
            if not self.targeted:
                y = y.detach().clone().long().to(self.device)
            else:
                y = self.get_target_label(x, y)

        if not self.targeted:
            acc = self.get_logits(x).max(1)[1] == y
        else:
            acc = self.get_logits(x).max(1)[1] != y

        startt = time.time()
        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                _, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                output_curr = self.get_logits(adv_curr)
                if not self.targeted:
                    acc_curr = output_curr.max(1)[1] == y_to_fool
                else:
                    acc_curr = output_curr.max(1)[1] != y_to_fool
                ind_curr = (acc_curr == 0).nonzero().squeeze()

                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                if self.verbose:
                    print(
                        "restart {} - robust accuracy: {:.2%}".format(
                            counter, acc.float().mean()
                        ),
                        "- cum. time: {:.1f} s".format(time.time() - startt),
                    )

        return adv




# QueryAttack ##########################################################
class QueryNet():
    def __init__(self, sampler, victim_name, surrogate_names, use_square_plus, use_square, use_nas, eps,
                 batch_size):
        self.surrogate_names = surrogate_names
        self.use_surrogate_attacker = self.surrogate_names != []
        self.use_square_plus = use_square_plus
        self.use_square = use_square
        assert (self.use_surrogate_attacker and self.use_square_plus and self.use_square) or \
               (self.use_surrogate_attacker and not self.use_square_plus and self.use_square) or \
               (self.use_surrogate_attacker and not self.use_square_plus and not self.use_square) or \
               (not self.use_surrogate_attacker and self.use_square_plus and not self.use_square) or \
               (not self.use_surrogate_attacker and not self.use_square_plus and self.use_square)
        self.eps = eps
        self.use_nas = use_nas

        self.train_loss_thres = 2 if ('easydl' not in victim_name) else 0.025  # easydl outputs prob instead of logits
        self.batch_size = batch_size
        self.victim_name = victim_name
        self.square_plus_max_trial = 50
        self.surrogate_train_iter = [30, 100, 1500]
        # stop training if the training loss < self.train_loss_thres after self.surrogate_train_iter[0] batches or anycase after self.surrogate_train_iter[1]
        # if the surrogate is not externally trained and save in ./pretrained by query pairs from the first two queries
        self.save_surrogate = True
        self.save_surrogate_path = sampler.result_dir + '/srg'
        os.makedirs(self.save_surrogate_path)

        self.sampler = sampler
        self.generator = PGDGeneratorInfty(int(batch_size / 2))
        self.square_attack = self.square_attack_linfty
        self.surrogates = []
        os.makedirs('query_attack_sub/pretrained', exist_ok=True)
        gpus = torch.cuda.device_count()
        num_class = self.sampler.label.shape[1]

        self.use_nas_layers = [10, 6, 8, 4, 12, 14]
        if self.sampler.data.shape[1] == 1: self.use_nas_layers = [int(x / 2) for x in
                                                                   self.use_nas_layers]  # use smaller search space for MNIST
        self.loaded_trained_surrogates_on_past_queries = []
        self.surrogate_save_paths = []
        for i, surrogate_name in enumerate(surrogate_names):
            if not self.use_nas:
                self.surrogates.append(Surrogate(surrogate_name, num_class=num_class, softmax='easydl' in victim_name,
                                                 gpu_id=0 if gpus == 1 else i % (gpus - 1) + 1))
            else:
                self.surrogates.append(
                    NASSurrogate(init_channels=16, layers=self.use_nas_layers[i], num_class=num_class,
                                 n_channels=self.sampler.data.shape[1], softmax='easydl' in victim_name,
                                 gpu_id=0 if gpus == 1 else i % (gpus - 1) + 1))

            save_info = 'query/pretrained/netSTrained_{}_{}_{}.pth'.format(surrogate_name, victim_name,
                                                                     0) if not self.use_nas else \
                'query/pretrained/netSTrained_NAS{}_{}_latest.pth'.format(self.use_nas_layers[i], self.victim_name)

            self.surrogate_save_paths.append(save_info)
            if os.path.exists(save_info):
                self.surrogates[i].load(save_info)
                self.loaded_trained_surrogates_on_past_queries.append(True)
            else:
                self.loaded_trained_surrogates_on_past_queries.append(False)

        self.num_attacker = len(surrogate_names) + int(use_square_plus) + int(use_square)
        self.attacker_eva_weights = [1] * len(surrogate_names) + [0, 0]
        if self.sampler.data.shape[1] == 1: self.attacker_eva_weights[-1] = 0

        if num_class == 1000:
            self.eva_weights_zero_threshold = 20  # set evaluation to zero if the denominator is small
        elif self.sampler.data.shape[1] == 1:
            self.eva_weights_zero_threshold = 2
        else:
            self.eva_weights_zero_threshold = 10

    def pseudo_gaussian_pert_rectangles(self, x, y):
        delta = np.zeros([x, y])
        x_c, y_c = x // 2 + 1, y // 2 + 1
        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
            max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2
            counter2[0] -= 1
            counter2[1] -= 1
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        return delta

    def meta_pseudo_gaussian_pert(self, s):
        delta = np.zeros([s, s])
        n_subsquares = 2
        if n_subsquares == 2:
            delta[:s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s)
            delta[s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
            if np.random.rand(1) > 0.5: delta = np.transpose(delta)

        elif n_subsquares == 4:
            delta[:s // 2, :s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, :s // 2] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
            delta[:s // 2, s // 2:] = self.pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice(
                [-1, 1])
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        return delta

    def square_attack_l2(self, x_curr, x_best_curr, deltas, is_candidate_maximizer, min_val, max_val, p, **kwargs):
        c, h, w = x_curr.shape[1:]
        n_features = c * h * w
        s = max(int(round(np.sqrt(p * n_features / c))), 3)
        if s % 2 == 0: s += 1
        s2 = s + 0

        ### window_1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = np.zeros(x_curr.shape)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

        ### window_2
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = np.zeros(x_curr.shape)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0

        ### compute total norm available
        curr_norms_window = np.sqrt(
            np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
        curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = np.sqrt(np.sum((deltas * mask_2) ** 2, axis=(2, 3), keepdims=True))

        ### create the updates
        new_deltas = np.ones([x_curr.shape[0], c, s, s])
        new_deltas = new_deltas * self.meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
        old_deltas = deltas[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
                np.maximum(self.eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
        deltas[~is_candidate_maximizer, :, center_h_2:center_h_2 + s2,
        center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
        deltas[~is_candidate_maximizer, :, center_h:center_h + s, center_w:center_w + s] = new_deltas[
                                                                                               ~is_candidate_maximizer, ...] + 0  # update window_1

        x_new = x_curr + deltas / np.sqrt(np.sum(deltas ** 2, axis=(1, 2, 3), keepdims=True)) * self.eps
        x_new = np.clip(x_new, min_val, max_val)
        return x_new, deltas

    def square_attack_linfty(self, x_curr, x_best_curr, deltas, is_candidate_maximizer, min_val, max_val, p, **kwargs):
        c, h, w = x_curr.shape[1:]
        n_features = c * h * w
        s = int(round(np.sqrt(p * n_features / c)))
        s = min(max(s, 1), h - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        if isinstance(deltas, torch.Tensor):
            deltas = deltas.cpu().numpy() 
        if isinstance(x_curr, torch.Tensor):
            x_curr = x_curr.cpu().numpy() 
        if isinstance(x_best_curr, torch.Tensor):
            x_best_curr = x_best_curr.cpu().numpy() 
        # 确保 deltas, x_curr 和 x_best_curr 不为 None
        if deltas is None or x_curr is None or x_best_curr is None:
            raise ValueError("deltas, x_curr, or x_best_curr is None after conversion to NumPy")

        deltas[~is_candidate_maximizer, :, center_h:center_h + s, center_w:center_w + s] = np.random.choice([-self.eps, self.eps], size=[c, 1, 1])

        # judge overlap
        for i_img in range(x_best_curr.shape[0]):
            if is_candidate_maximizer[i_img]: continue
            center_h_tmp, center_w_tmp, s_tmp = center_h, center_w, s
            while np.sum(np.abs(np.clip(
                    x_curr[i_img, :, center_h_tmp:center_h_tmp + s_tmp, center_w_tmp:center_w_tmp + s_tmp] +
                    deltas[i_img, :, center_h_tmp:center_h_tmp + s_tmp, center_w_tmp:center_w_tmp + s_tmp],
                    min_val, max_val) -
                                x_best_curr[i_img, :, center_h_tmp:center_h_tmp + s_tmp,
                                center_w_tmp:center_w_tmp + s_tmp])
                         < 10 ** -7) == c * s * s:
                s_tmp = int(round(np.sqrt(p * n_features / c)))
                s_tmp = min(max(s_tmp, 1), h - 1)
                center_h_tmp, center_w_tmp = np.random.randint(0, h - s_tmp), np.random.randint(0, w - s_tmp)
                deltas[i_img, :, center_h_tmp:center_h_tmp + s_tmp,
                center_w_tmp:center_w_tmp + s_tmp] = np.random.choice([-self.eps, self.eps], size=[c, 1, 1])
        result = np.clip(x_curr + deltas, min_val, max_val)
        if deltas is None:
            raise ValueError("None deltas square_linfty")
        if result is None:
            raise ValueError("None result square_linfty")
        return result, deltas

    def square_attacker(self, x_curr, x_best_curr, **kwargs):
        x_next, _ = self.square_attack(x_curr, x_best_curr, x_best_curr - x_curr, np.zeros(x_best_curr.shape[0], dtype=np.bool), **kwargs)
        return x_next

    def square_plus_attacker(self, x_curr, x_best_curr, **kwargs):
        is_candidate_maximizer = np.zeros(x_best_curr.shape[0], dtype=np.bool)
        deltas = x_best_curr - x_curr
        deltas = deltas.cpu().numpy()  ########
        x_curr = x_curr.cpu().numpy()
        x_best_curr = x_best_curr.cpu().numpy()
        for i in range(
                self.square_plus_max_trial):  # retry random search for a maximum of self.square_plus_max_trial times
            x_next, deltas = self.square_attack(x_curr, x_best_curr, deltas, is_candidate_maximizer, **kwargs)
            is_candidate_maximizer = self.sampler.judge_potential_maximizer(x_next)
            if np.sum(is_candidate_maximizer) == x_best_curr.shape[0]: break
        return x_next

    def surrogate_attacker(self, x_curr, x_best_curr, y_curr, attacker_id, targeted):  # , **kwargs):
        assert attacker_id < len(self.surrogate_names)
        os.makedirs(self.sampler.result_dir + '/log', exist_ok=True)
        log_file_path = '%s/log/surrogate%d_train.log' % (self.sampler.result_dir, attacker_id)
        for i in range(self.surrogate_train_iter[1] if self.loaded_trained_surrogates_on_past_queries[attacker_id] else
                       self.surrogate_train_iter[2]):
            train_loss = self.surrogates[attacker_id].train(attacker_id, self.sampler, self.batch_size, i,
                                                            log_file_path=log_file_path)
            if train_loss < self.train_loss_thres and i > self.surrogate_train_iter[
                0]: break  # train surrogate until convergence
            if not self.loaded_trained_surrogates_on_past_queries[attacker_id]:
                self.surrogates[attacker_id].save(self.surrogate_save_paths[attacker_id])
                # if the surrogates are not pretrained on first 2 iteration queries, conduct thorough training and save it for later usage
                self.loaded_trained_surrogates_on_past_queries[attacker_id] = True

        iter_trained = 0
        while 1:
            if not self.use_nas:
                save_info = '{}/netSTrained_{}_{}_{}.pth'.format(self.save_surrogate_path,
                                                                 self.surrogate_names[attacker_id], self.victim_name,
                                                                 iter_trained)
            else:
                save_info = '{}/netSTrained_NAS{}_{}_{}.pth'.format(self.save_surrogate_path,
                                                                    self.use_nas_layers[attacker_id], self.victim_name,
                                                                    iter_trained)
            if not os.path.exists(save_info): break
            iter_trained += 1
        if self.save_surrogate: self.surrogates[attacker_id].save(save_info)
        # FGSM attack
        self.x_new_tmp[attacker_id] = self.generator(x_best_curr, x_curr, self.eps, self.surrogates[attacker_id], y_curr, targeted=targeted)

    def surrogate_attacker_multi_threading(self, x_curr, x_best_curr, y_curr, targeted, **kwargs):
        threads = []  # train and attack via different surrogates simultaneously
        self.x_new_tmp = [0 for _ in range(len(self.surrogate_names))]
        for attacker_id in range(len(self.surrogate_names)):
            threads.append(threading.Thread(target=self.surrogate_attacker,
                                            args=(x_curr, x_best_curr, y_curr, attacker_id, targeted)))
        for attacker_id in range(len(self.surrogate_names)): threads[attacker_id].start()
        for attacker_id in range(len(self.surrogate_names)):
            if threads[attacker_id].isAlive(): threads[attacker_id].join()
        return self.x_new_tmp

    def yield_candidate_queries(self, x_curr, x_best_curr, y_curr, **kwargs):
        if max(self.attacker_eva_weights) == self.attacker_eva_weights[-2]:  # max(w_{1~n}) < w_{n+1}, w_{n+2} < w_{n+1}
            x_new_candidate = []
            if self.use_square_plus:  x_new_candidate.append(self.square_plus_attacker(x_curr, x_best_curr, **kwargs))
            if self.use_square:      x_new_candidate.append(self.square_attacker(x_curr, x_best_curr, **kwargs))
            return x_new_candidate
        elif max(self.attacker_eva_weights) == self.attacker_eva_weights[-1]:  # max(w_{1~n}) < w_{n+1} < w_{n+2}
            return [self.square_attacker(x_curr, x_best_curr, **kwargs)]
        else:  # max(w_{1~n}) > w_{n+1}
            x_new_candidate = self.surrogate_attacker_multi_threading(x_curr, x_best_curr, y_curr, **kwargs)
            if self.use_square_plus:
                x_new_candidate.append(self.square_plus_attacker(x_curr, x_best_curr, **kwargs))
            elif self.use_square:
                x_new_candidate.append(self.square_attacker(x_curr, x_best_curr, **kwargs))
            return x_new_candidate

    def forward(self, x_curr, x_best_curr, y_curr, get_surrogate_loss, **kwargs):
        x_new_candidate = self.yield_candidate_queries(x_curr, x_best_curr, y_curr, **kwargs)
        if len(x_new_candidate) == 1:
            return x_new_candidate[0], None  # max(w_{1~n}) < w_{n+1} < w_{n+2}, use square only
        else:
            loss_candidate = []  # num_attacker * num_sample
            for attacker_id in range(len(x_new_candidate)):
                loss_candidate_for_one_attacker = []
                for evaluator_id in range(len(self.surrogate_names)):
                    loss_candidate_for_one_attacker.append(
                        get_surrogate_loss(self.surrogates[evaluator_id], x_new_candidate[attacker_id], y_curr.cpu().numpy()) *
                        self.attacker_eva_weights[evaluator_id]
                    )
                loss_candidate.append(sum(loss_candidate_for_one_attacker) / len(loss_candidate_for_one_attacker))
            loss_candidate = np.array(loss_candidate)

            x_new_index = np.argmin(loss_candidate, axis=0)  # a, selected attacker IDs
            x_new = np.zeros(x_curr.shape)  # x^q
            for attacker_id in range(len(x_new_candidate)):
                attacker_index = x_new_index == attacker_id
                x_new[attacker_index] = x_new_candidate[attacker_id][attacker_index]
        return x_new, x_new_index

    def backward(self, idx_improved, x_new_index, **kwargs):  # 应该用的是cpu，因为update_buffer有np计算
        if self.use_surrogate_attacker:
            if self.use_square_plus or self.use_square:
                save_only = max(self.attacker_eva_weights) == self.attacker_eva_weights[-1]
                self.sampler.update_buffer(save_only=save_only, **kwargs)
                if x_new_index is not None and self.use_square_plus and not save_only: self.sampler.update_lipschitz()
            else:
                self.sampler.update_buffer(save_only=False, **kwargs)  # FGSM and Square+ require to update buffer
        elif self.use_square_plus:  # Square+ only, no surrogates
            print("use square+")
            self.sampler.update_buffer(save_only=False, **kwargs)
            self.sampler.update_lipschitz()  # only do this for square+
        elif self.use_square:  # Square only, no surrogates
            self.sampler.update_buffer(save_only=True, **kwargs)

        if x_new_index is None:
            return None  # Square, do nothing

        elif max(self.attacker_eva_weights) == self.attacker_eva_weights[-2]:  # Square+, Square
            print("use square+, square")
            assert x_new_index.max() == 1 and self.use_square_plus and self.use_square  # only valid when they are both adopted
            attacker_selected = [0 for _ in range(len(self.surrogate_names))]
            for attacker_id in range(x_new_index.max() + 1):

                attacker_index = x_new_index == attacker_id
                attacker_selected.append(np.mean(attacker_index))
                attacker_id_real = attacker_id + len(self.surrogate_names)

                if np.sum(attacker_index) < self.eva_weights_zero_threshold:
                    self.attacker_eva_weights[attacker_id_real] = 0  # 21.5.4  few samples fail to judge the eva_weights
                else:
                    self.attacker_eva_weights[attacker_id_real] = np.sum(idx_improved[attacker_index]) / np.sum(
                        attacker_index)

        else:
            # (1) FGSM, Square+ in QueryNet (A = {FGSM, Square+, Square})
            # (2) FGSM, Square if we do not include Square+ (A = {FGSM, Square})
            # (3) FGSM if A = {FGSM}
            # (4) Square+ if A = {Square+}
            assert x_new_index.max() in [len(self.surrogate_names) - 1, len(self.surrogate_names)]
            attacker_selected = []
            for attacker_id in range(x_new_index.max() + 1):

                attacker_index = x_new_index == attacker_id
                if x_new_index.max() == len(self.surrogate_names) - 1 or attacker_id != x_new_index.max():
                    #                                           (3) or attacker_id is not the last, no need to handle attacker_selected exceptionally
                    attacker_id_real = attacker_id
                    attacker_selected += [np.mean(attacker_index)]
                    if attacker_id == x_new_index.max(): attacker_selected += [0, 0]  # (3)
                elif self.use_square_plus:  # attacker_id is the last, (1), (4)
                    attacker_id_real = attacker_id
                    attacker_selected += [np.mean(attacker_index), 0]  # no Square in these cases
                else:  # attacker_id is the last, (2)
                    attacker_id_real = attacker_id + 1  # no Square+ in this case, so the last index of x_new is actually for Square
                    attacker_selected += [0, np.mean(attacker_index)]

                if np.sum(attacker_index) < self.eva_weights_zero_threshold:
                    self.attacker_eva_weights[attacker_id_real] = 0
                else:
                    self.attacker_eva_weights[attacker_id_real] = np.sum(idx_improved[attacker_index]) / np.sum(
                        attacker_index)

        field_names = ['ATTACK']
        if not self.use_nas:
            field_names += self.surrogate_names
        else:
            field_names += ['NASlayer' + str(l) for l in self.use_nas_layers[:len(self.surrogate_names)]]
        field_names += ['Square+', 'Square']
        tb = pt.PrettyTable()
        tb.field_names = field_names
        width = {}
        for i, field_name in enumerate(field_names):
            if i: width[field_name] = 11
        tb._min_width = width
        tb._max_width = width
        tb.add_row(['WEIGHT'] + ['%.3f' % x for x in self.attacker_eva_weights])
        tb.add_row(['CHOSEN'] + ['%.3f' % x for x in attacker_selected])
        return str(tb)


class PGDGeneratorInfty():
    def __init__(self, max_batch_size):
        self.device = torch.device('cuda:0')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.max_batch_size = max_batch_size

    def _call(self, img, lbl, surrogate, epsilon, targeted):
        # img : B * H * W * C  0~1 np.float32 array
        img = img.to(surrogate.device)
        img.requires_grad = True
        lbl = torch.Tensor(lbl).to(surrogate.device)

        alpha = epsilon * 2
        num_iter = 1

        momentum_grad = 0
        for i in range(num_iter):
            img.requires_grad = True
            loss = self.criterion(surrogate(img, no_grad=False).float(), lbl.argmax(dim=-1))
            surrogate.surrogate.zero_grad()
            loss.backward()
            grad = img.grad.data
            momentum_grad += grad
            img = img + alpha * momentum_grad.sign()  # maximum attack step: FGSM
        return img.to(self.device)

    def __call__(self, img, ori, epsilon, surrogate, lbl, return_numpy=True, targeted=False):
        # img : B * H * W * C  0~1 np.float32 array
        # return: B * H * W * C  np.float32 array   /   B * C * H * W  0~1  torch.Tensor
        # CPU
        torch.cuda.empty_cache()
        img, ori = torch.Tensor(img), torch.Tensor(ori)
        batch_size = min([self.max_batch_size, img.shape[0]])
        if batch_size < self.max_batch_size:
            adv = self._call(img, lbl, surrogate, epsilon, targeted=targeted)
        else:
            batch_num = int(img.shape[0] / batch_size)
            if batch_size * batch_num != int(img.shape[0]): batch_num += 1
            adv = self._call(img[:batch_size], lbl[:batch_size], surrogate, epsilon, targeted=targeted)
            for i in range(batch_num - 1):
                new_adv = torch.cat((adv,
                                     self._call(img[batch_size * (i + 1):batch_size * (i + 2)],
                                                lbl[batch_size * (i + 1):batch_size * (i + 2)],
                                                surrogate, epsilon, targeted=targeted)), 0)
                adv = new_adv

        adv = torch.min(torch.max(adv, ori - epsilon), ori + epsilon)
        adv = torch.clamp(adv, 0.0, 1.0)
        if return_numpy:
            return adv.detach().cpu().numpy()
        else:
            return adv


class PGDGenerator2():
    def __init__(self, max_batch_size):
        self.device = torch.device('cuda:0')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.max_batch_size = max_batch_size

    def _call(self, img, ori, lbl, surrogate, epsilon, targeted):
        # img : B * H * W * C  0~1 np.float32 array
        img = img.to(surrogate.device)
        img.requires_grad = True
        lbl = torch.Tensor(lbl).to(surrogate.device)

        alpha = epsilon * 2

        loss = self.criterion(surrogate(img, no_grad=False), lbl.argmax(dim=-1))
        surrogate.surrogate.zero_grad()
        loss.backward()
        grad = img.grad.data
        img = img + alpha * grad / \
              torch.norm(grad.reshape(grad.shape[0], -1), dim=1, p=2, keepdim=True).reshape(-1).repeat(grad.shape[1],
                                                                                                       grad.shape[2],
                                                                                                       grad.shape[3],
                                                                                                       1).permute(3, 0,
                                                                                                                  1, 2)
        return img.to(self.device)

    def __call__(self, img, ori, epsilon, surrogate, lbl, return_numpy=True, targeted=False):
        # img : B * H * W * C  0~1 np.float32 array
        # return: B * H * W * C  np.float32 array   /   B * C * H * W  0~1  torch.Tensor
        # CPU
        torch.cuda.empty_cache()
        img, ori = torch.Tensor(img), torch.Tensor(ori)
        batch_size = min([self.max_batch_size, img.shape[0]])
        if batch_size < self.max_batch_size:
            adv = self._call(img, ori, lbl, surrogate, epsilon, targeted=targeted)
        else:
            batch_num = int(img.shape[0] / batch_size)
            if batch_size * batch_num != int(img.shape[0]): batch_num += 1
            adv = self._call(img[:batch_size], ori[:batch_size], lbl[:batch_size], surrogate, epsilon,
                             targeted=targeted)
            for i in range(batch_num - 1):
                # print(i, batch_num, end='\r')
                new_adv = torch.cat((adv,
                                     self._call(img[batch_size * (i + 1):batch_size * (i + 2)],
                                                ori[batch_size * (i + 1):batch_size * (i + 2)],
                                                lbl[batch_size * (i + 1):batch_size * (i + 2)],
                                                surrogate, epsilon, targeted=targeted)), 0)
                adv = new_adv

        per = adv - ori
        adv = ori + per / \
              torch.norm(per.reshape(per.shape[0], -1), dim=1, p=2, keepdim=True).reshape(-1).repeat(per.shape[1],
                                                                                                     per.shape[2],
                                                                                                     per.shape[3],
                                                                                                     1).permute(3, 0, 1,
                                                                                                                2) * epsilon
        adv = torch.clamp(adv, 0.0, 1.0)
        if return_numpy:
            return adv.detach().numpy()
        else:
            return adv


###### also queryattack ################################################
def p_selection(p_init, it, num_iter):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / num_iter * 10000)
    if   10 < it <= 50:       return p_init / 2
    elif 50 < it <= 200:      return p_init / 4
    elif 200 < it <= 500:     return p_init / 8
    elif 500 < it <= 1000:    return p_init / 16
    elif 1000 < it <= 2000:   return p_init / 32
    elif 2000 < it <= 4000:   return p_init / 64
    elif 4000 < it <= 6000:   return p_init / 128
    elif 6000 < it <= 8000:   return p_init / 256
    elif 8000 < it <= 10000:  return p_init / 512
    else:                     return p_init



class QueryAttack():
    r"""
    Arguments:
        model_names: default='resnext101_32x8d', type=str, [inception_v3, mnasnet1_0, resnext101_32x8d] for ImageNet.
        num_x: type=int, default=10000, number of samples for evaluation.
        num_srg: type=int, default=0, number of surrogates.
        use_nas: action='store_true', use NAS to train the surrogate.
        use_square_plus: action='store_true', use Square+.
        p_init: type=float, default=0.05, hyperparameter of Square, the probability of changing a coordinate.
        run_times: type=int, default=1, repeated running time.
        l2_attack: action='store_true', perform l2 attack
        num_iter: type=int, default=10000, maximum query times.
        gpu: type=str, default='1', GPU number(s).
    Examples:
        >>> x_test, y_test, logits_clean = attack.QueryAttack.get_xylogits(model, model_names='resnext101_32x8d', num_x=10000, eps=8/255)
        >>> attack = attack.QueryAttack(model, eps=8/255, iter=10000)
        >>> adv_images = attack(model, x, y, logits_clean)     # images---x, labels---y
    """

    def __init__ (self, model, eps=8/255, num_iter=10000, num_x=10000):
        # super().__init__("QueryAttack", model)
        # some parameters
        self.eps = eps
        self.num_iter = num_iter
        self.gpu = '1'
        self.num_srg = 3 # number of surrogates
        self.dataset = 'image_net'
        self.use_square_plus = True # use Square+
        self.use_nas = False # don't use NAS to train the surrogate
        self.batch_size = 16 # batch_size = 100 if model_name != 'resnext101_32x8d' else 32
        self.p_init = 0.05 # hyperparameter of Square, the probability of changing a coordinate
        self.seed = 1 # for random number
        self.num_x = num_x # default: 10000, number of samples for evaluation

    
    def get_xylogits(self, model_names):
        # model_names: default: 'resnext101_32x8d', '[inception_v3, mnasnet1_0, resnext101_32x8d] for ImageNet'
        np.random.seed(self.seed)
        if self.use_nas: assert self.num_srg > 0
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        for model_name in model_names.split(','):
            if model_name in ['inception_v3', 'mnasnet1_0', 'resnext101_32x8d']:
                dataset = 'imagenet'
            else:
                raise ValueError('Invalid Victim Name!')

        # imagenet 我们先假设model_name只有一个，即只选一个网络 #################################
        assert (not self.use_nas), 'NAS is not supported for ImageNet for resource concerns'
        if not (self.eps == 12.75):  # (args.l2_attack and args.eps == 5) or (not args.l2_attack and eps == 12.75)
            print('Warning: not using default eps in the paper, which is linfty=12.75 for ImageNet.')
        model = VictimImagenet(model_name, batch_size=self.batch_size)
        x_test, y_test = load_imagenet(self.num_x, model)

        logits_clean = model(x_test)
        corr_classified = logits_clean.argmax(1) == y_test.argmax(1)
        print('Clean accuracy: {:.2%}'.format(np.mean(corr_classified)) + ' ' * 40)
        y_test = dense_to_onehot(y_test.argmax(1), n_cls=1000)

        return x_test[corr_classified], y_test[corr_classified], logits_clean[corr_classified], model
        # x_test = torch.tensor(x_test[corr_classified], dtype=torch.float32)
        # y_test = torch.tensor(y_test[corr_classified], dtype=torch.float32)
        # logits_clean = torch.tensor(logits_clean[corr_classified], dtype=torch.float32)

        # return x_test, y_test, logits_clean

    

    def __call__(self, model, x, y, logits_clean):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        np.random.seed(self.seed)
        min_val, max_val = 0, 1
        c, h, w = x.shape[1:]
        # n_features = c * h * w

        x_best = np.clip(x + np.random.choice([-self.eps, self.eps], size=[x.shape[0], c, 1, w]), min_val, max_val)
        x_best_init = x_best.copy()

        # logits = model(torch.tensor(x_best).float())

        # 显存不够的话，分批计算logits
        num_batches = (len(x_best) + self.batch_size - 1) // self.batch_size  # 计算总批次数，确保最后的不足一个批次的数据也会被处理
        logits_list = []
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(x_best))  # 确保索引不超出数据范围
            x_batch = x_best[start_idx:end_idx]
            if x_batch.shape[0] == 0:
                continue
            # x_batch = torch.from_numpy(x_batch).float().to('cuda:0')
            with torch.no_grad():
                logits_batch = model(x_batch)
            logits_list.append(logits_batch)
            del x_batch
            del logits_batch
            torch.cuda.empty_cache()
        # logits = torch.cat(logits_list, dim=0)
        logits = np.concatenate(logits_list, axis=0)

        loss_min = get_margin_loss(torch.from_numpy(y).to('cuda:0'), torch.from_numpy(logits).to('cuda:0'))# logits)
        n_queries = np.ones(x.shape[0]) * 2  # have queried with original samples and stripe samples

        surrogate_names = ['DenseNet121', 'ResNet50', 'DenseNet169', 'ResNet101', 'DenseNet201', 'VGG19'][:self.num_srg] # surrogates if not using nas
        result_path = get_time() + f'_{self.dataset}_ResNet101' + \
            ('_linfty') + \
            f'_eps{round(self.eps*255, 2)}' + \
            ('_Eval' if self.num_srg != 0 else '') + \
            ('_Sqr+' if self.use_square_plus else '') + \
            (f'_NAS{self.num_srg}' if self.use_nas else ('_'+'-'.join(surrogate_names) if len(surrogate_names) != 0 else ''))
        print(result_path)

        log = Logger('')
        logger = LoggerUs(result_path)
        os.makedirs(result_path + '/log', exist_ok=True)
        log.reset_path(result_path + '/log/main.log')
        metrics_path = logger.result_paths['base'] + '/log/metrics'

        sampler = DataManager(x, logits_clean, self.eps, result_dir=result_path, loss_init=get_margin_loss(y, logits_clean))
        # x_best_cpu = x_best.cpu().numpy()  # 转换为 NumPy 数组 .detach
        # logits_cpu = logits.cpu().numpy()
        sampler.update_buffer(x_best, logits, loss_min, logger, targeted=False, data_indexes=None, margin_min=loss_min)
        sampler.update_lipschitz()
        querynet = QueryNet(sampler, 'ResNet101', surrogate_names, self.use_square_plus, True, self.use_nas, self.eps, self.batch_size)

        def get_surrogate_loss(srgt, x_adv, y_ori): # for transferability evaluation in QueryNet's 2nd forward operation
            if x_adv.shape[0] <= self.batch_size:  return get_margin_loss(y_ori, srgt(torch.Tensor(x_adv)).cpu().detach().numpy())
            batch_num = int(x_adv.shape[0]/self.batch_size)
            if self.batch_size * batch_num != int(x_adv.shape[0]): batch_num += 1
            loss_value = get_margin_loss(y_ori[:self.batch_size], srgt(torch.Tensor(x_adv[:self.batch_size])).cpu().detach().numpy())
            for i in range(batch_num-1):
                new_loss_value = get_margin_loss(y_ori[self.batch_size*(i+1):self.batch_size*(i+2)], srgt(torch.Tensor(x_adv[self.batch_size*(i+1):self.batch_size*(i+2)])).cpu().detach().numpy())
                loss_value = np.concatenate((loss_value, new_loss_value), axis=0)
                # loss_value = torch.cat((loss_value, new_loss_value), dim=0)
                del new_loss_value
            return loss_value

        time_start = time.time()
        metrics = np.zeros([self.num_iter, 7])
        for i_iter in range(self.num_iter):
            # focus on unsuccessful AEs
            idx_to_fool = loss_min > 0
            # x_curr, x_best_curr, y_curr, loss_min_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool], loss_min[idx_to_fool]
            x_curr = torch.from_numpy(x[idx_to_fool]).to(device)
            x_best_curr = torch.from_numpy(x_best[idx_to_fool]).to(device)
            y_curr = torch.from_numpy(y[idx_to_fool]).to(device)
            loss_min_curr = loss_min[idx_to_fool]

            # QueryNet's forward propagation
            x_q, a = querynet.forward(x_curr.float(), x_best_curr.float(), y_curr.float(), get_surrogate_loss, min_val=min_val, max_val=max_val, p=p_selection(self.p_init, i_iter, self.num_iter), targeted=False)

            # query
            # logits = model(x_q)
            # 显存不够的话，分批计算logits
            num_batches = (len(x_q) + self.batch_size - 1) // self.batch_size  # 计算总批次数，确保最后的不足一个批次的数据也会被处理
            logits_list = []
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(x_q))  # 确保索引不超出数据范围
                x_batch = x_q[start_idx:end_idx]
                if x_batch.shape[0] == 0:
                    continue
                # x_batch = torch.from_numpy(x_batch).float().to('cuda:0')
                with torch.no_grad():
                    logits_batch = model(x_batch)
                logits_list.append(logits_batch)
                del x_batch
                del logits_batch
                torch.cuda.empty_cache()
            # logits = torch.cat(logits_list, dim=0)
            logits = np.concatenate(logits_list, axis=0)

            loss = get_margin_loss(y_curr,  torch.from_numpy(logits).to('cuda:0'))# logits)
            idx_improved = loss < loss_min_curr
            loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
            idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
            x_best[idx_to_fool] = idx_improved * x_q+ ~idx_improved * x_best_curr.cpu().numpy()
            n_queries[idx_to_fool] += 1
            # if i_iter == self.num_iter-1:
            ppath = "/iter_{}".format(i_iter+1)
            os.makedirs(logger.result_paths['adv'] + ppath, exist_ok=True)
            save_imgs(x_best, indexes=np.arange(1000), result_path_adv=logger.result_paths['adv'] + ppath)

            # QueryNet's backward propagation
            if a is not None:
                a = a.astype(np.int32)
            # message = querynet.backward(idx_improved, a, data_indexes=np.where(idx_to_fool)[0], margin_min=loss_min, img_adv=x_q.astype(np.float32), lbl_adv=logits_clean, loss=loss, logger=logger, targeted=False)
            message = querynet.backward(idx_improved, a, data_indexes=np.where(idx_to_fool)[0], 
            margin_min=loss_min, img_adv=x_q, lbl_adv=logits, loss=loss, logger=logger, targeted=False)
            if a is not None:
                print(' '*80, end='\r')
                log.print(message)
                querynet.sampler.save(i_iter)
            

            # logging
            acc_corr = (loss_min > 0.0).mean()
            mean_nq_all, mean_nq = np.mean(n_queries), np.mean(n_queries[loss_min <= 0])
            median_nq_all, median_nq = np.median(n_queries)-1, np.median(n_queries[loss_min <= 0])-1
            avg_loss = np.mean(loss_min)
            elapse = time.time() - time_start
            msg = '{}: Acc={:.2%}, AQ_suc={:.2f}, MQ_suc={:.1f}, AQ_all={:.2f}, MQ_all={:.1f}, ALoss_all={:.2f}, |D|={:d}, Time={:.1f}s'.\
                format(i_iter + 1, acc_corr, mean_nq, median_nq, mean_nq_all, median_nq_all, avg_loss, querynet.sampler.clean_sample_indexes[-1], elapse)
            log.print(msg) # if 'easydl' not in model.arch else msg + ', query=%d' % model.query)
            metrics[i_iter] = [acc_corr, mean_nq, median_nq, mean_nq_all, median_nq_all, avg_loss, elapse]
            np.save(metrics_path, metrics)
            if acc_corr == 0: break

            # 测试每一轮的x_best的准确率
            batch_size = 16
            model.eval()
            correct = 0
            num_batches = (len(x_best) + batch_size - 1) // batch_size  # 计算总批次数
            output_list = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(x_best))
                x_ = x_best[start_idx:end_idx]
                if x_.shape[0] == 0:
                    continue
                # x_ = torch.from_numpy(x_).float().to('cuda')
                with torch.no_grad():
                    output_batch = model(x_)
                output_list.append(output_batch)
                del x_
                del output_batch
                torch.cuda.empty_cache()
            # outputs = torch.cat(output_list, dim=0)
            outputs = np.concatenate(output_list, axis=0)

            _, predicted = torch.max(torch.tensor(outputs), 1)
            predicted = predicted.cpu()
            y_test = torch.argmax(torch.tensor(y).to(torch.int32), dim=1)
            total = y_test.size(0)
            correct += (predicted == y_test).sum().item()
            # 打印维度
            print("Predicted shape:", predicted.shape)
            print("y_test shape:", y_test.shape)

            # equal_pairs = predicted == y_test 
            # correct = torch.sum(equal_pairs).item()  # 使用 .item() 来获取数值
            accuracy = 100 * correct / total
            msg_xbest = '{}: Accuracy of x_best in each iteration: {:.2f}%, correct = {}, total = {}'.\
                format(i_iter + 1, accuracy, correct, total)
            log.print(msg_xbest)

            pixel_diff = x_best - x_best_init
            # 计算差值的平均值
            mean_diff = np.mean(np.abs(pixel_diff))
            print("每个像素差值的平均值:", mean_diff.item())
            
        torch.cuda.empty_cache()
        # os.makedirs(logger.result_paths['adv']+"/x_q", exist_ok=True)
        # save_imgs(x_q, indexes=np.arange(1000), result_path_adv=logger.result_paths['adv']+"./x_q")

        return x_best_init, x_best


'''
    def QueryAttack(args):
        query_net, eps, seed, l2_attack, num_iter, p_init, num_srg, use_square_plus, use_nas = \
            (args.eps / 255 if not args.l2_attack else args.eps), (args.seed if args.seed != -1 else args.run_time), \
            args.query_net, args.l2_attack, args.num_iter, args.p_init, args.num_srg, args.use_square_plus, args.use_nas
        np.random.seed(args.seed)

        if args.use_nas: assert args.num_srg > 0
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        log = Logger('')

        for model_name in args.query_net.split(','):
            if model_name in ['inception_v3', 'mnasnet1_0', 'resnext101_32x8d']:         dataset = 'imagenet'
            else: raise ValueError('Invalid Victim Name!')

            # imagenet
            assert (not args.use_nas), 'NAS is not supported for ImageNet for resource concerns'
            if not ((args.l2_attack and args.eps == 5) or (not args.l2_attack and args.eps == 12.75)):
                print('Warning: not using default eps in the paper, which is l2=5 or linfty=12.75 for ImageNet.')
            batch_size = 100 if model_name != 'resnext101_32x8d' else 32
            model = VictimImagenet(model_name, batch_size=batch_size)
            x_test, y_test = load_imagenet(args.num_x, model)

            logits_clean = model(x_test)
            corr_classified = logits_clean.argmax(1) == y_test.argmax(1)
            print('Clean accuracy: {:.2%}'.format(np.mean(corr_classified)) + ' ' * 40)
            y_test = dense_to_onehot(y_test.argmax(1), n_cls=10 if dataset != 'imagenet' else 1000)
            for run_time in range(args.run_times):
                attack(model, x_test[corr_classified], y_test[corr_classified], logits_clean[corr_classified], dataset, batch_size, run_time, args, log)
'''

