import os
import typing
import random
import torch
import matplotlib.pyplot as plt
import detectron2
import detectron2.data
import detectron2.engine
import detectron2.config
import detectron2.model_zoo
import detectron2.evaluation
import detectron2.utils.logger
import detectron2.utils.visualizer


from .. import core


def to_detectron(name: str, collection: core.SceneCollection, directory: str = None):
    """Insert a scene collection into the detectron singletons."""
    dictionaries = [
        {
            "file_name": scene.filepath(directory=directory),
            "height": scene.dimensions.height,
            "width": scene.dimensions.width,
            "image_id": idx,
            "annotations": [
                {
                    "bbox": ann.x1y1x2y2(),
                    "category_id": scene.categories.index(ann.category),
                    "bbox_mode": detectron2.structures.BoxMode.XYXY_ABS,
                    "segmentation": [ann.points.ravel().astype("float32").tolist()],
                }
                for ann in scene.annotations
            ],
        }
        for idx, scene in enumerate(collection)
    ]
    try:
        detectron2.data.DatasetCatalog.remove(name)
    except KeyError:
        pass
    try:
        detectron2.data.MetadataCatalog.remove(name)
    except KeyError:
        pass
    detectron2.data.DatasetCatalog.register(name, lambda: dictionaries)
    detectron2.data.MetadataCatalog.get(name).set(
        thing_classes=[c.name for c in collection.categories]
    )


def visualize(name: str, n: int = 3):
    """Build a visual using detectron's utilities."""
    dictionaries = random.sample(detectron2.data.DatasetCatalog.get(name), n)
    fig, axs = plt.subplots(ncols=len(dictionaries), figsize=(15, 15))
    for ax, d in zip(axs if isinstance(axs, list) else [axs], dictionaries):
        image = core.utils.read(d["file_name"])
        ax.imshow(
            detectron2.utils.visualizer.Visualizer(
                image, scale=1.0, metadata=detectron2.data.MetadataCatalog.get(name)
            )
            .draw_dataset_dict(d)
            .get_image()
        )
    return fig, axs


def score(
    trainer: detectron2.engine.TrainerBase,
):
    """Score a model using inference_on_dataset."""
    trainer.test(
        trainer.cfg,
        trainer.model,
        evaluators=[
            detectron2.evaluation.COCOEvaluator(
                trainer.cfg.DATASETS.TEST[0], output_dir=trainer.cfg.OUTPUT_DIR
            )
        ],
    )


def build_trainer(
    train,
    validation: str,
    base_cfg: str,
    weights: str = None,
    iterations=300,
    roi_batch_size=128,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Build a model trainer."""
    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(base_cfg)
    cfg.DATASETS.TRAIN = (train,)
    cfg.DATASETS.TEST = (validation,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = weights
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = iterations
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(
        detectron2.data.MetadataCatalog.get(train).thing_classes
    )
    cfg.MODEL.DEVICE = device
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = detectron2.engine.DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer, cfg


def instances_to_annotations(
    instances, categories: core.Categories
) -> typing.List[core.Annotation]:
    """Convert detectron2 instances type to annotations"""
    return [
        core.Annotation(categories[category], x1=x1, y1=y1, x2=x2, y2=y2, score=score)
        for (x1, y1, x2, y2), score, category in zip(
            instances.pred_boxes.tensor.numpy(),
            instances.scores.numpy(),
            instances.pred_classes.numpy(),
        )
    ]


def infer(model, images, categories: core.Categories):
    """Run inference on a batch of images."""
    return [
        instances_to_annotations(instances, categories)
        for instances in model(
            [{"image": image} for image in images], categories=categories
        )
    ]
