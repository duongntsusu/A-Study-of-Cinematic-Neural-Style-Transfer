# A-Study-of-Cinematic-Neural-Style-Transfer

**1. Introduction:**

This is a PyTorch implementation of my research project in the field of Neural Style Transfer (NST). In this research project, I investigate how Neural Style Transfer can be adapted to render static images in cinematic styles, focusing on three styles associated with three directors: Tim Burton, Stanley Kubrick, and Wes Anderson. Using a curated film-based style dataset - where each cinematic style is represented through a collection of images - I implement two separate experiments, both of which use Conditional Instance Normalization, to transfer cinematic styles to static images.

Some implementation details are inspired by code in this repo: <https://github.com/tyui592/A_Learned_Representation_For_Artistic_Style>.

**2. Explain the two experiments in-depth:**

- For the first experiment: during each epoch, within each batch of content images, a random style image and its style index are sampled from the style dataset to generate stylized images for each reference content image in the batch. The style loss is computed using the Gram matrices of the feature maps extracted from the stylized outputs and the randomly sampled style image across chosen layers for style reconstruction.
- For the second experiment: Given the specified layers for stylistic extraction, with every style image that corresponds to a certain style j, a loss network is called to extract stylistic maps from that image. Then, iterating over every layer where those feature maps are captured, the function adds up the Gram matrices of the features corresponding to that layer and averages them by the number of style images of style j to compute an average style representation. I hypothesize that by adopting the approach of computing the average Gram matrices across each director's set of cinematic frames, the style transfer model can capture the general stylistic characteristics of each director's cinematography. 
   
**3. Datasets:**

- Content: Kaggle's COCO WikiArt dataset
- Style: I used a self-curated film-based style dataset, where each cinematic style associated with one director is captured through a collection of cinematic frames from that director's filmography. Particularly, in the case of Wes Anderson, frames are extracted from the films The Grand Budapest Hotel (2014), Moonrise Kingdom (2012), Asteroid City (2023), The Phoenician Scheme (2025), and The Wonderful Story of Henry Sugar (2023). For Tim Burton, the live-action films selected are Beetlejuice (1988), Batman Returns (1992), Edward Scissorhands (1990), and Alice in Wonderland (2010). In the case of Stanley Kubrick, movies like 2001: A Space Odyssey (1968), The Shining (1980), Clockwork Orange (1972), and Eyes Wide Shut (1999) are chosen.  

**4. Project structure:**

- Main: Contains the code for the training function 
- Network_Architecture: Contains the code for the model's architecture 
- Loss: Functions that compute the Gram Matrix of a given feature map, style loss, content loss, and total variation loss.
- First_experiment: Contains the code for my implementation of the first experiment
- Second_experiment: Contains the code for my implementation of the second experiment
- Reconstructions:  Contains all content and style reconstruction experiments for analyzing learned feature representations.

**5. References:**
For our implementation of Conditional Instance Normalization, we adopt the same network architecture as in the paper "A Learned Representation for Artistic Style" (reference:  https://arxiv.org/abs/1610.07629)



