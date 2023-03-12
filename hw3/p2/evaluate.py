import os
import json
from collections import defaultdict
from argparse import ArgumentParser
from PIL import Image

import clip
import torch
import language_evaluation


def readJSON(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except:
        return None


def getGTCaptions(annotations):
    img_id_to_name = {}
    for img_info in annotations["images"]:
        img_name = img_info["file_name"].replace(".jpg", "")
        img_id_to_name[img_info["id"]] = img_name

    img_name_to_gts = defaultdict(list)
    for ann_info in annotations["annotations"]:
        img_id = ann_info["image_id"]
        # img_name = 000000249720 (contains 0)
        img_name = img_id_to_name[img_id]
        img_name_to_gts[img_name].append(ann_info["caption"])
    return img_name_to_gts


class CIDERScore:
    def __init__(self):
        self.evaluator = language_evaluation.CocoEvaluator(coco_types=["CIDEr"])
        # self.gts = gts
        # self.precompute_gt()

    def __call__(self, predictions, gts):
        """
        Input:
            predictions: dict of str
            gts:         dict of list of str
        Return:
            cider_score: float
        """
        # Collect predicts and answers
        predicts = []
        answers = []
        for img_name in predictions.keys():
            predicts.append(predictions[img_name])
            answers.append(gts[img_name])
        
        # Compute CIDEr score
        results = self.evaluator.run_evaluation(predicts, answers)
        return results['CIDEr']

    def precompute_gt(self):
        self.answers = []
        for img_name in self.gts.keys():
            self.answers.append(self.gts[img_name])

    def compute_cider(self, predictions):
        predicts = []
        for img_name in self.gts.keys():
            predicts.append(predictions[img_name])
        results = self.evaluator.run_evaluation(predicts, self.answers)
        return results['CIDEr']

    #cider_score = CIDERScore()(predictions, gts)

class CLIPScore:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def precompute_gt(self):
        pass

    def __call__(self, predictions, images_root):
        """
        Input:
            predictions: dict of str
            images_root: str
        Return:
            clip_score: float
        """
        total_score = 0.

        for img_name, pred_caption in predictions.items():
            image_path = os.path.join(images_root, f"{img_name}.jpg")
            image = Image.open(image_path).convert("RGB")
            total_score += self.getCLIPScore(image, pred_caption)
        return total_score / len(predictions)

    def getCLIPScore(self, image, caption):
        """
        This function computes CLIPScore based on the pseudocode in the slides.
        Input:
            image: PIL.Image
            caption: str
        Return:
            cilp_score: float
        """
        w = 2.5
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize(caption).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

        cos_sim = torch.nn.functional.cosine_similarity(img_features, text_features).item()
        clip_score = w * max(cos_sim, 0)
        return clip_score