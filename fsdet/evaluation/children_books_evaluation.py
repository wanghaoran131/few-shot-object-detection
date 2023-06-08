import json
import os
import copy
from collections import OrderedDict
import detectron2.utils.comm as comm
from fsdet.evaluation.evaluator import DatasetEvaluator
from fsdet.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.data import MetadataCatalog
from fsdet.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import logging
import itertools


class CHILDRENBOOKSEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None): # initial needed variables
        self._distributed = distributed
        self._dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(dataset_name)
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning("Metadata '{}' does not have attribute 'json_file'!".format(dataset_name))

        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._class_names = self._metadata.things_classes
        self._base_classes = self._metadata.base_classes
        self._novel_classes = self._metadata.novel_classes
        self._do_evaluation = len(self._metadata.base_classes) > 0 and "annotations" in self._metadata.dataset

    def reset(self): # reset predictions
        self._predictions = []
        self._children_books_results = []

    def process(self, inputs, outputs): # prepare predictions for evaluation
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self): # evaluate predictions
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))
            
            if not comm.is_main_process():
                return {}
            
        if len(self._predictions) == 0:
            self._logger.warning("[CHILDRENBOOKSEvaluator] Did not receive valid predictions.")
            return {}
        
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)
        
        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        
        return copy.deepcopy(self._results)


    def _eval_predictions(self): # evaluate predictions
        '''
        Evaluate predictions.
        '''
        self._logger.info("Preparing results for COCO format ...")
        self._children_books_results = list(itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "class_mapping"):
            # using reverse mapping
            reverse_id_mapping = {v: k for k, v in self._metadata.class_mapping.items()}
            for result in self._children_books_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]] + 1
        
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "children_books_instances_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._children_books_results))
                f.flush()
        
        if not self._do_evaluation:
            self._logger.warning("Annotations are not available for evaluation!")
            return
        
        self._logger.info("Evaluating predictions ...")
        res = _evaluate_predictions_on_children_books(
            self._metadata.dataset, 
            self._children_books_results, 
            "bbox", 
            catIds=self._metadata.base_classes
        )
        self._results["bbox"] = res

    
def _evaluate_predictions_on_children_books(
    children_books_gt, children_books_results, iou_type, catIds=None
):
    '''
    Evaluate predictions on children books.
    '''
    metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

    logger = logging.getLogger(__name__)

    if len(children_books_results) == 0:
        logger.warning("No predictions from the model! Set scores to -1")
        return {metric: -1 for metric in metrics}
    
    from lvis import LVISEval, LVISResults

    children_books_results = LVISResults(children_books_gt, children_books_results)
    children_books_eval = LVISEval(children_books_gt, children_books_results, iou_type)
    children_books_eval.run()
    children_books_eval.print_results()

    results = children_books_eval.get_results()
    results = {metric: float(results[metric]) for metric in metrics}
    logger.info(
        "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
    )
    return results