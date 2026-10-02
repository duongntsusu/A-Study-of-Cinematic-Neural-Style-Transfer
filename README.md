# Representing Cinematic Style with Neural Style Transfer

**1. Introduction:**

This is a PyTorch implementation of my research project in Neural Style Transfer, for which I authored and published a 13-page research paper in the National High School Journal of Science (NHSJS) <https://nhsjs.com/2026/representing-cinematic-style-with-neural-style-transfer-a-case-study-of-films-by-wes-anderson-and-tim-burton>

In this research project, under the guidance of Dr. Mariel Werner from UC Berkeley, I investigated how Neural Style Transfer can be adapted to render static images in cinematic styles, focusing on styles associated with Tim Burton and Wes Anderson's filmography. Using a curated film-based style dataset, I implemented two experiments that employ the multi-style transfer framework of Dumoulin and colleagues (2016) <https://arxiv.org/abs/1610.07629> to compare the average Gram-matrix style representation of multiple images with that of a single reference image. 

Some implementation details are inspired by code in this repo: <https://github.com/tyui592/A_Learned_Representation_For_Artistic_Style>.

   
**2. Datasets:**

- Training content dataset: Kaggle's COCO WikiArt dataset
- Validation/testing content dataset: ImageNet dataset
- Style: I used a self-curated film-based style dataset, where each cinematic style associated with one director is captured through a collection of cinematic frames from that director's filmography. Cinematic frames are obtained from [FilmGrab] (https://film-grab.com/)

**3. Project structure:**

- Main: Contains the code for the training function 
- Network_Architecture: Contains the code for the model's architecture 
- Loss: Functions that compute the Gram Matrix of a given feature map, style loss, content loss, and total variation loss.
- First_experiment: Contains the code for my implementation of the first experiment
- Second_experiment: Contains the code for my implementation of the second experiment
- Reconstructions:  Contains all content and style reconstruction experiments for analyzing learned feature representations.



