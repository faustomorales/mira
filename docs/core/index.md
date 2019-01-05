# Core Functions

.. toctree::
   scenes
   annotations
   selections
   images

The core functions in `mira` are for handling data. Object detection data requires working with images and annotations together in a sensible way.

First, we have to track what types of objects we're detecting, the `AnnotationConfiguration` object stores the list of categories that are permitted as `AnnotationCategory` objects. 

Second, we need to assign detections to images. To do this, we use the `Scene` object, which consists of an `Image`, a list of `Annotation` and an `AnnotationConfiguration`. Each `Annotation` contains a `Selection` (which describes the area of the image where the annotation points) and an `AnnotationCategory` to categorize it.

Finally, we need to group scenes together using a `SceneCollection`, which also has an `AnnotationConfiguration` assigned to it (the collection has functions for verifying the integrity of its collected scenes). Recognizing this can be confusing, a graph of these relationships is provided below.


.. mermaid::
   graph TD
        SC[SceneCollection] --> |has many| S[Scene]
        SC --> |has an| ACF[AnnotationConfiguration]
        S --> |has an| ACF
        S --> |has an| IM[Image]
        S --> |has many| A[Annotation]
        ACF --> |has many| AC[AnnotationCategory]
        A --> |has an| AC
        A --> |has a| SEL[Selection]
