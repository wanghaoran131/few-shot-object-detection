import argparse
import json
import os
import random

CAT_NAMES = ['acorn', 'axe', 'backpack', 'badger', 'bag', 'barrel', 'basket', 'bear', 'bed', 'bee', 'bell', 'bench', 'bird', 'birdcage', 'boar', 'boat', 'book', 'bottle', 'bow', 'bowl', 'box', 'bridge', 'broom', 'brush', 'bucket', 'building', 'butterfly', 'camel', 'campfire', 'candle', 'cane', 'cannon', 'car', 'cat', 'cello', 'chair', 'clock', 'couch', 'cow', 'cradle', 'crown', 'cup', 'curtain', 'deer', 'diningTable', 'dog', 'doghouse', 'donkey', 'door', 'dragon', 'drum', 'egg', 'elephant', 'ermine', 'feather', 'female', 'fence', 'fireplace', 'fish', 'fishingRod', 'flag', 'flower', 'flute', 'fox', 'frog', 'glasses', 'globe', 'goat', 'gun', 'hammer', 'hat', 'hedgehog', 'helmet', 'horse', 'hotAirBalloon', 'inkpot', 'insect', 'jackal', 'jar', 'jug', 'kettle', 'kite', 'knife', 'ladder', 'lamp', 'lifebuoy', 'lion', 'lizard', 'lobster', 'male', 'map', 'marmot', 'melon', 'monkey', 'moon', 'musicSheet', 'nest', 'net', 'painting', 'paintingStand', 'pan', 'pear', 'pen', 'penguin', 'piano', 'pickaxe', 'pig', 'pineapple', 'pipe', 'plant', 'plate', 'pot', 'pottedPlant', 'rabbit', 'rake', 'rat', 'rhino', 'sausage', 'saw', 'scale', 'scissors', 'scorpion', 'seal', 'shark', 'sheep', 'shield', 'shovel', 'sieve', 'skate', 'snail', 'snake', 'spear', 'spoon', 'sportsBall', 'squirrel', 'star', 'stool', 'stroller', 'suitcase', 'sun', 'sunflower', 'sword', 'teachingBoard', 'teapot', 'tent', 'tie', 'tiger', 'train', 'tree', 'trumpet', 'tub', 'turtle', 'umbrella', 'vase', 'violin', 'wagon', 'walnut', 'weight', 'whip', 'windmill', 'window', 'wineGlass', 'wolf', 'zebra']
base_classes_names = ['male', 'female', 'bird', 'hat', 'tree', 'dog', 'horse', 'building', 'chair', 'window', 'cat', 'flower', 'diningTable', 'basket', 'boat', 'book', 'curtain', 'cup']
base_classes = [CAT_NAMES.index(c) for c in base_classes_names]
novel_classes = [c for c in range(164) if c not in base_classes]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/Children_Books_COCO/train/_annotations.coco.json",
        help="path to the annotation file",
    )
    parser.add_argument(
        "--shots", type=int, default=3, help="number of shots"
    )
    args = parser.parse_args()
    return args


def get_shots(args):
    data = json.load(open(args.data, "r"))
    ann = data["annotations"]

    anno_cat = {i: [] for i in range(164)}
    for a in ann:
        anno_cat[a["category_id"] - 1].append(a)

    anno = [] # get args.shots annos for each category and concat them, include all if less than args.shots
    for i, c in enumerate(CAT_NAMES):
        if len(anno_cat[i]) < args.shots:
            shots = anno_cat[i]
        else:
            shots = random.sample(anno_cat[i], args.shots)
        anno.extend(shots)

    new_data = {
        "info": data["info"],
        "licenses": data["licenses"],
        "categories": data["categories"],
        "images": data["images"],
        "annotations": anno,
    }

    save_path = os.path.join("datasets/CB_split", "3_shots.json")
    with open(save_path, "w") as f:
        json.dump(new_data, f)


if __name__ == "__main__":
    random.seed(421)

    args = parse_args()
    get_shots(args)
