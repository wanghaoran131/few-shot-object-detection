from fsdet.evaluation.evaluator import DatasetEvaluator
from detectron2.data import MetadataCatalog
from fsdet.utils.file_io import PathManager
import torch
import logging


class CHILDRENBOOKSEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir=None): # initial needed variables
        self._dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(dataset_name)
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)


    def reset(self): # reset predictions
        self._predictions = []

    def process(self, inputs, outputs): # prepare predictions for evaluation
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                prediction["instances"] = output["instances"]
            self._predictions.append(prediction)

    def evaluate(self): # evaluate predictions
        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APf"]

        results = evaluate_predictions(self._predictions)
        return {
            "AP": results["AP"],
            "AP50": results["AP50"],
            "AP75": results["AP75"],
        }