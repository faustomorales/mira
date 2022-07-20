Core Functions
**************

.. toctree::
    scenes
    annotations

The core functions in `mira` are for handling data. Object detection data requires working with images and annotations together in a sensible way.

First, we have to track what types of objects we're detecting, the `Categories` object stores the list of categories that are permitted as `Category` objects.

Second, we need to assign detections to images. To do this, we use the `Scene` object, which consists of an `Image`, a list of `Annotation` and an `Categories`. Each `Annotation` contains the bounding box and an `Category` to categorize it.

Finally, we need to group scenes together using a `SceneCollection`, which also has an `Categories` assigned to it (the collection has functions for verifying the integrity of its collected scenes). Recognizing this can be confusing, a graph of these relationships is provided below.

.. mermaid::

    graph TD
        SC[SceneCollection] --> |has many| S[Scene]
        SC --> |has an| ACF[Categories]
        S --> |has an| ACF
        S --> |has an| IM[Image]
        S --> |has many| A[Annotation]
        ACF --> |has many| AC[Category]
        A --> |has an| AC
