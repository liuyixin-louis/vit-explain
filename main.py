import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device="cuda:2") * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


import timm
import torch
from torch import nn
net = timm.create_model("vit_base_patch16_384", pretrained=True)
net.head = nn.Linear(net.head.in_features, 10)
# module problem
state_dict = torch.load("/home/yila22/prj/vision-transformers-cifar10/checkpoint/vit_timm-4-ckpt.t7", map_location="cpu")["model"]
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
from ut import *
device = 'cuda:2'
net = net.to(device)

import torchvision
from torchvision import transforms
import torch
# model = torch.hub.load('facebookresearch/deit:main', 
# 'deit_tiny_patch16_224', pretrained=True)

# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform_test = transforms.Compose([
    transforms.Resize(384),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = net

class Averager(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
from tqdm import tqdm
smooth_model = Smooth(model, 10, 4/255)


# poisoning case with defense

# get model
# get dataset  
# loop over dataset
# for data inside the poison set
# get the original attention mask
# poison the data with Adversarial Attack with epsilon
# 1. denoise poisoned image with pretrained diffusion model with epsilon
# 2. conduct random smoothing on the denoised image with delta of R computed using (2*delta ?)
# get the attention mask after poisoning
# get the metrics: robust accuracy, top-k mask overlap under pertubed

# Loss is CE
from torch import nn
criterion = nn.CrossEntropyLoss()


# poisoning case with defense

# get model
# get dataset  
# loop over dataset
# for data inside the poison set
# get the original attention mask
# poison the data with Adversarial Attack with epsilon
# 1. denoise poisoned image with pretrained diffusion model with epsilon
# 2. conduct random smoothing on the denoised image with delta of R computed using (2*delta ?)
# get the attention mask after poisoning
# get the metrics: robust accuracy, top-k mask overlap under pertubed


import numpy as np
# poisoning case without defense

# get model
# get dataset  
# loop over dataset
net.eval()
test_loss = 0
adv_test_loss=0
correct = 0
correct_adv = 0
total = 0

from vit_rollout import VITAttentionRollout
rollout = VITAttentionRollout(model, discard_ratio=0.9, head_fusion='max')
masks = []
masks_after_poison = []
topk_index_orignal = []
topk_index_after = []
topk_overlap = []
topk_overlap_value = Averager()
robust_acc = 0

import torchattacks
torch.backends.cudnn.deterministic = True
atk = torchattacks.PGD(model, eps=4/255, alpha=2/255, steps=5)
atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

# 
for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
    if total > 100:
        break
    with torch.no_grad():
        
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            # % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # print("clean accuracy: ", 100.*correct/total)

        # get the original attention mask
        images_to_list = [inputs[i] for i in range(inputs.shape[0])]
        
        for image in images_to_list:
            image = image.unsqueeze(0)
            attention_mask = rollout(image)
            masks.append(attention_mask)
    
    # poison the data with Adversarial Attack
    inputs, targets = inputs.to(device), targets.to(device)
    adv_images = atk(inputs, targets)
    
    # we have defense here
    # using smoothing technique for prediction
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        prediction = []
        for i in range(adv_images.shape[0]):
            pred = smooth_model.predict(adv_images[i].to(device=device), n=100, alpha=0.05, batch_size = 32)
            if pred == -1:
                prediction.append(np.random.choice(10))
            else:
                prediction.append(pred)
        prediction = torch.tensor(prediction).to(device)
        total += targets.size(0)
        correct_adv += prediction.eq(targets).sum().item()
        # print("robust accuracy: ", 100.*correct_adv/total)
        robust_acc = 100.*correct_adv/total
    
    
    # get the attention mask after poisoning
    with torch.no_grad():
        N_sample = 16
        with torch.no_grad():
            adv_images_to_list = [adv_images[i] for i in range(inputs.shape[0])]
            for adv_image_i in adv_images_to_list:
                adv_image_i = adv_image_i.unsqueeze(0)
                mask_after = np.zeros((N_sample, 24, 24))
                # mean voting
                for Ni in range(N_sample):
                    adv_image_i_perturbed = adv_image_i + torch.randn_like(adv_image_i) * 4/255
                    adv_image_i_perturbed = torch.clamp(adv_image_i_perturbed, 0, 1)
                    attention_mask_after = rollout(adv_image_i_perturbed)
                    mask_after[Ni] = attention_mask_after
                masks_after_poison.append(np.mean(mask_after, axis = 0))

    # get the metrics: robust accuracy,
    # with torch.no_grad():
    #     adv_images, targets = adv_images.to(device), targets.to(device)
    #     outputs_after = model(adv_images)
    #     adv_loss = criterion(outputs_after, targets)
    #     adv_test_loss += adv_loss.item()
    #     _, predicted = outputs_after.max(1)
    #     total += targets.size(0)
    #     correct_adv += predicted.eq(targets).sum().item()
    #     # print("robust accuracy: ", 100.*correct_adv/total)
    #     robust_acc = 100.*correct_adv/total
    
    #  top-k mask overlap under pertubed
    K = 5
    bs = inputs.shape[0]
    mask_original = np.concatenate(masks[-bs:], axis=0)
    mask_after_poison = np.concatenate(masks_after_poison[-bs:], axis=0)
    mask_original = mask_original.reshape(bs, -1)
    mask_after_poison = mask_after_poison.reshape(bs,-1)
    mask_after_poison = torch.tensor(mask_after_poison)
    mask_original = torch.tensor(mask_original)
    topk_index_orignal.append(torch.topk(mask_original, k=K, dim=1).indices)
    topk_index_after.append(torch.topk(mask_after_poison, k=K, dim=1).indices)
    topk_overlap.append(torch.mean((torch.sum(topk_index_orignal[-1] == topk_index_after[-1], dim=1).float())/K).item())
    
    # print("top-k mask overlap under pertubed: ", topk_overlap[-1])
    topk_overlap_value.update(topk_overlap[-1], bs)
    
    # break
print("final robust accuracy: ", robust_acc)
print("final top-k mask overlap under pertubed: ", topk_overlap_value.avg)