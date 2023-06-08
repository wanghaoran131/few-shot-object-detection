import argparse
import json
import os

CAT_NAMES = ['acorn', 'axe', 'backpack', 'badger', 'bag', 'barrel', 'basket', 'bear', 'bed', 'bee', 'bell', 'bench', 'bird', 'birdcage', 'boar', 'boat', 'book', 'bottle', 'bow', 'bowl', 'box', 'bridge', 'broom', 'brush', 'bucket', 'building', 'butterfly', 'camel', 'campfire', 'candle', 'cane', 'cannon', 'car', 'cat', 'cello', 'chair', 'clock', 'couch', 'cow', 'cradle', 'crown', 'cup', 'curtain', 'deer', 'diningTable', 'dog', 'doghouse', 'donkey', 'door', 'dragon', 'drum', 'egg', 'elephant', 'ermine', 'feather', 'female', 'fence', 'fireplace', 'fish', 'fishingRod', 'flag', 'flower', 'flute', 'fox', 'frog', 'glasses', 'globe', 'goat', 'gun', 'hammer', 'hat', 'hedgehog', 'helmet', 'horse', 'hotAirBalloon', 'inkpot', 'insect', 'jackal', 'jar', 'jug', 'kettle', 'kite', 'knife', 'ladder', 'lamp', 'lifebuoy', 'lion', 'lizard', 'lobster', 'male', 'map', 'marmot', 'melon', 'monkey', 'moon', 'musicSheet', 'nest', 'net', 'painting', 'paintingStand', 'pan', 'pear', 'pen', 'penguin', 'piano', 'pickaxe', 'pig', 'pineapple', 'pipe', 'plant', 'plate', 'pot', 'pottedPlant', 'rabbit', 'rake', 'rat', 'rhino', 'sausage', 'saw', 'scale', 'scissors', 'scorpion', 'seal', 'shark', 'sheep', 'shield', 'shovel', 'sieve', 'skate', 'snail', 'snake', 'spear', 'spoon', 'sportsBall', 'squirrel', 'star', 'stool', 'stroller', 'suitcase', 'sun', 'sunflower', 'sword', 'teachingBoard', 'teapot', 'tent', 'tie', 'tiger', 'train', 'tree', 'trumpet', 'tub', 'turtle', 'umbrella', 'vase', 'violin', 'wagon', 'walnut', 'weight', 'whip', 'windmill', 'window', 'wineGlass', 'wolf', 'zebra']
base_classes_names = ['male', 'female', 'bird', 'hat', 'tree', 'dog', 'horse', 'building', 'chair', 'window', 'cat', 'flower', 'diningTable', 'basket', 'boat', 'book', 'curtain', 'cup']
base_classes = [CAT_NAMES.index(c)+1 for c in base_classes_names]
novel_classes = [c for c in range(1, 165) if c not in base_classes]
print(base_classes)
print(novel_classes)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="Children_Books_COCO/annotations/train_annotations.json",
        help="path to the annotation file",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="children_books_split",
        help="path to the save directory",
    )
    args = parser.parse_args()
    return args


def split_annotation(args):
    with open(args.data) as fp:
        ann_train = json.load(fp)


    for name, classes in [("freq", base_classes), ("rare", novel_classes)]:
        ann_s = {
            "info": ann_train["info"],
            "images": ann_train['images'],
            "categories": ann_train["categories"],
            "licenses": ann_train["licenses"],
            "annotations": ann_train["annotations"],
        }

        ids = classes
        print(ids)
        ann_s["categories"] = [cat for cat in ann_train["categories"] if cat["id"] in ids]
        ann_s["annotations"] = [ann for ann in ann_train["annotations"] if ann["category_id"] in ids]

        # get image_id ffom ann_s["annotations"], then get image from ann_train["images"]
        ann_s["images"] = [img for img in ann_train["images"] if img["id"] in set([ann["image_id"] for ann in ann_s["annotations"]])]
        save_path = os.path.join(args.save_dir, "children_books_train_{}.json".format(name))
        print("Saving {} annotations to {}.".format(name, save_path))
        with open(save_path, "w") as fp:
            json.dump(ann_s, fp)


if __name__ == "__main__":
    args = parse_args()
    split_annotation(args)
