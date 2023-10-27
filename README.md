# SegmentUKI project
Segment Understanding Knowledgeable Imaging system

#### Credentials: Vladislav Kulikov, Vladislav Urzhumov

## Project objective
SegmentUKI project's task is to provide models for translation of segmentation maps into real images, which is the task of conditional image synthesis.
Current github repository aims to utilize generative potential of known concepts found in SOTA systems for segmentation map TO image task, such as:
* [NVidia's SPADE {1}](https://arxiv.org/pdf/1903.07291.pdf)
* [pix2pix HD {2}](https://arxiv.org/pdf/1711.11585.pdf)

> **A note on copyrights**
> Here and below in the current markdown file, the provided images are images taken from the listed sources {1} and {2}. Credentials for the images can be found at the links above.
> We are not authors or distributors of these images to any extent, thus for copyrights please address the authors of the papers above.
> 
> Any additional material without explicitly noted source is the result of intellectial work made by or summarized by the contributors of the current github repository.
> MIT license for content in the current repository is present in the main directory of master branch.
> It is highly recommended to read it if you intend to cite, copy or distribute the content of the current repository. We are grateful for your deliberate actions.

Links listed lead to research papers with concept descriptions and are elaborated more below in this documentation.

>|**Note:** Architectures of the systems above are a subject to change and set up to achieve better understanding of segmentation maps inside the models.|

## Key concepts of the project
The main so-to-say USP of SegmentUKI is to provide real knowledgeable imaging via segmentation map understanding. Understanding of segmentation maps includes the following parts:

#### 1. Semantic segment partition of the map
> Classical segmentation understanding
Semantic segment partition understanding simply means clear division of classes represented by semantically different segments. Shortly, treat semantically different segments as different classes.
Classical segmentation maps provide extensive information about semantic classes of objects from the map, thus classical pix2pix models are able to understand it starting at early training stages.
![Semantic segment partition example from {1}](https://github.com/v-like-engine/segmentuki/assets/57713513/d1478c94-d22a-43cb-bee4-d52403382820)


#### 2. Instance segment partition of the map
> Application for more complex segmentation task
Instance segment partition understanding leads the model to higher level of image representation of the maps: a whole block of semantic car class is transformed into several car class instances, each with it's own predicted or pre-highlighted boundaries. At this point, instance boundary map extraction is utilized to achieve understanding of this exact aspect.
![Instance boundary map from {2}](https://github.com/v-like-engine/segmentuki/assets/57713513/e06cee4b-461a-4105-a1a0-fe339e78eef0)


#### 3. Spatial features understanding
> Information extracted from locations of segments
Spatial features are what really important in any computer vision task. However, spatial information is corrupted or lost once the classical normalization is applied to a map or image.
SPADE {1} is a conditional trained layer of spatially-adaptive normalization which is a type of normalization used to preserve spatial information while performing the normalization with all of it benefits.
Park et al. {1} claim that SPADE improves the performance of pix2pix models compared to usual normalization layers.
SegmentUKI will utilize benefits of SPADE in some of project's modules.

![SPADE illustration from {1}](https://github.com/v-like-engine/segmentuki/assets/57713513/b95247ae-1b58-499b-a178-036ad582c15d)


#### 4. Layering
> Overlapping of segments lead to layering
Some segments may overlap on the segmentation map and should be treated as layering. Solid layers are claimed to be visible throughout more transparent once.
This is the level of understanding beyond the classical segment2image synthesis task. Condition passed to pix2pix is a subject to be set up by preprocessing, and that approach is how we address the problem


## Progress

The project is started at **Autumn 2003** and is currently in progress.
Implementation of pix2pix model and preprocessing are done yet are still a subject to change or edit.


## Structure

Project is structured and architecture is maintained with good code practices. Later on this section will be embraced to navigate users and possible new collaborators explaining positions and modules of the code


## How to start

Current section will contain a guidance on how-to-use the system once we will have the actual one-fits-all running routine or executable files. Stay tuned!
