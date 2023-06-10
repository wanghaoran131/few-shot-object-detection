# import json
# import os
# import copy
# from collections import OrderedDict
# import detectron2.utils.comm as comm
# from fsdet.evaluation.evaluator import DatasetEvaluator
# from fsdet.evaluation.coco_evaluation import instances_to_coco_json
# from detectron2.data import MetadataCatalog
# from fsdet.utils.file_io import PathManager
# from detectron2.utils.logger import create_small_table
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# import torch
# import logging
# import itertools
# from lvis import LVIS
# from lvis import LVISEval, LVISResults


# class CHILDRENBOOKSEvaluator(DatasetEvaluator):
#     def __init__(self, dataset_name, cfg, distributed, output_dir=None): # initial needed variables
#         self._distributed = distributed
#         self._dataset_name = dataset_name
#         self._metadata = MetadataCatalog.get(dataset_name)
#         self._output_dir = output_dir
#         self._cpu_device = torch.device("cpu")
#         self._logger = logging.getLogger(__name__)

#         if not hasattr(self._metadata, "json_file"):
#             self._logger.warning("Metadata '{}' does not have attribute 'json_file'!".format(dataset_name))

#         json_file = PathManager.get_local_path(self._metadata.json_file)
#         self._children_books_gt = LVIS(json_file)
#         self._class_names = self._metadata.thing_classes
#         self._base_classes = self._metadata.base_classes
#         self._novel_classes = self._metadata.novel_classes
#         self._do_evaluation = len(self._metadata.base_classes) > 0

#     def reset(self): # reset predictions
#         self._predictions = []
#         self._children_books_results = []

#     def process(self, inputs, outputs): # prepare predictions for evaluation
#         for input, output in zip(inputs, outputs):
#             prediction = {"image_id": input["image_id"]}
#             if "instances" in output:
#                 instances = output["instances"].to(self._cpu_device)
#                 prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
#             self._predictions.append(prediction)

#     def evaluate(self): # evaluate predictions
#         if self._distributed:
#             comm.synchronize()
#             self._predictions = comm.gather(self._predictions, dst=0)
#             self._predictions = list(itertools.chain(*self._predictions))
            
#             if not comm.is_main_process():
#                 return {}
            
#         if len(self._predictions) == 0:
#             self._logger.warning("[CHILDRENBOOKSEvaluator] Did not receive valid predictions.")
#             return {}
        
#         if self._output_dir:
#             PathManager.mkdirs(self._output_dir)
#             file_path = os.path.join(self._output_dir, "instances_predictions.pth")
#             with PathManager.open(file_path, "wb") as f:
#                 torch.save(self._predictions, f)
        
#         self._results = OrderedDict()
#         if "instances" in self._predictions[0]:
#             self._eval_predictions()
        
#         return copy.deepcopy(self._results)


#     def _eval_predictions(self): # evaluate predictions
#         '''
#         Evaluate predictions.
#         '''
#         self._logger.info("Preparing results for COCO format ...")
#         self._children_books_results = list(itertools.chain(*[x["instances"] for x in self._predictions]))

#         # unmap the category ids for COCO
#         if hasattr(self._metadata, "class_mapping"):
#             # using reverse mapping
#             reverse_id_mapping = {v: k for k, v in self._metadata.class_mapping.items()}
#             for result in self._children_books_results:
#                 result["category_id"] = reverse_id_mapping[result["category_id"]] + 1
        
#         if self._output_dir:
#             file_path = os.path.join(self._output_dir, "children_books_instances_predictions.json")
#             with PathManager.open(file_path, "w") as f:
#                 f.write(json.dumps(self._children_books_results))
#                 f.flush()
        
#         if not self._do_evaluation:
#             self._logger.warning("Annotations are not available for evaluation!")
#             return
        
#         self._logger.info("Evaluating predictions ...")

#         res = _evaluate_predictions_on_children_books(
#             self._children_books_gt, 
#             self._children_books_results, 
#             "bbox", 
#             catIds=self._metadata.base_classes
#         )
#         self._results["bbox"] = res

    
# def _evaluate_predictions_on_children_books(
#     children_books_gt, children_books_results, iou_type, catIds=None
# ):
#     '''
#     Evaluate predictions on children books.
#     '''
#     metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

#     logger = logging.getLogger(__name__)

#     if len(children_books_results) == 0:
#         logger.warning("No predictions from the model! Set scores to -1")
#         return {metric: -1 for metric in metrics}
    

#     children_books_results = LVISResults(children_books_gt, children_books_results)
#     children_books_eval = LVISEval(children_books_gt, children_books_results, iou_type)
#     children_books_eval.run()
#     children_books_eval.print_results()

#     results = children_books_eval.get_results()
#     results = {metric: float(results[metric]) for metric in metrics}
#     logger.info(
#         "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
#     )
#     return results

import contextlib
import copy
import io
import itertools
import json
import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import BoxMode
from detectron2.utils.logger import create_small_table
from fsdet.evaluation.evaluator import DatasetEvaluator
from fsdet.utils.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate


class CHILDRENBOOKSEvaluator(DatasetEvaluator):
    """
    Evaluate instance detection outputs using COCO's metrics and APIs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                    so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True):
                if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'"
            )

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        self._is_splits = (
            "all" in dataset_name
            or "base" in dataset_name
            or "novel" in dataset_name
        )
        # fmt: off
        self._base_classes = self._metadata.base_classes
        # self._base_classes = [
        #     8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
        #     36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        #     55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
        #     81, 82, 84, 85, 86, 87, 88, 89, 90,
        # ]
        # fmt: on

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"]
                )
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}
        print("coco_evaluation.py: evaluate", self._predictions)
        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions."
            )
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth"
            )
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions])
        )

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k
                for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[
                    result["category_id"]
                ]

        if self._output_dir:
            file_path = os.path.join(
                self._output_dir, "coco_instances_results.json"
            )
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        if self._is_splits:
            self._results["bbox"] = {}
            for split, classes, names in [
                ("all", None, self._metadata.get("thing_classes")),
                ("base", self._base_classes, self._metadata.get("base_classes")),
                ("novel", self._novel_classes, self._metadata.get("novel_classes")),
            ]:
                if (
                    "all" not in self._dataset_name
                    and split not in self._dataset_name
                ):
                    continue
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api,
                        self._coco_results,
                        "bbox",
                        classes,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res_ = self._derive_coco_results(
                    coco_eval,
                    "bbox",
                    class_names=names,
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b" + metric] = res_[metric]
                        elif split == "novel":
                            res["n" + metric] = res_[metric]
                self._results["bbox"].update(res)

            # add "AP" if not already in
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]
        else:
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    self._coco_results,
                    "bbox",
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(
                coco_eval,
                "bbox",
                class_names=self._metadata.get("thing_classes"),
            )
            self._results["bbox"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

        if coco_eval is None:
            self._logger.warn(
                "No predictions from the model! Set scores to -1"
            )
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100)
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type)
            + create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2] - 1

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)]
        )
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})

        # save per-category results into a json
        if self._output_dir:
            file_path = os.path.join(
                self._output_dir, "coco_instances_per_category_results.json"
            )
            with PathManager.open(file_path, "w") as f:
                json.dump(results_per_category, f)
        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


def _evaluate_predictions_on_coco(
    coco_gt, coco_results, iou_type, catIds=None
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if catIds is not None:
        coco_eval.params.catIds = catIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
