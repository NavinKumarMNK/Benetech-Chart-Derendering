# BeneTech - Making Graphs Accessible 

![Alt text](<assets/benetech.png>)

## Matcha 

> MatCha is a model that is trained using the Pix2Struct architecture1. It is a Visual Question Answering subset of the Pix2Struct architecture. It renders the input question on the image and predicts the answer.

> MatCha is a model that enhances visual language models’ capabilities in jointly modeling charts/plots and language data. It is trained using the Pix2Struct architecture and performs several pretraining tasks that cover plot deconstruction and numerical reasoning, which are key capabilities in visual language modeling.

> On standard benchmarks such as PlotQA and ChartQA, the MatCha model outperforms state-of-the-art methods by as much as nearly 20%1. This is a significant improvement over other VQA models. MatCha pretraining also transfers well to domains such as screenshots, textbook diagrams, and document figures, verifying the usefulness of MatCha pretraining on broader visual language tasks

![Alt text](<assets/matcha.png>)
## Procedure 

> Matcha (Pix2Struct) : To Derender Line, Vertical Bar, Horizontal Bar 

> For Dot & Scatter Plot : icevision Faster RCNN Model 

## Datasets Used

> Official Kaggle Competition Data : https://www.kaggle.com/competitions/benetech-making-graphs-accessible/data

> Extra Generated Data : https://www.kaggle.com/datasets/brendanartley/benetech-extra-generated-data <br>
    - Deplot Training Format

> IDCAR 2023 : https://www.kaggle.com/datasets/sambitmukherjee/b-m-g-a-extra-data <br> 
    - Additional images have been picked from the ICDAR 2023 Competition on Harvesting Answers and Raw Tables from Infographics (CHART-Infographics) website: https://chartinfo.github.io/ <br>
    - Annotations have been created using Make Sense (makesense.ai) & Roboflow, and through exploratory data analysis of the Benetech - Making Graphs Accessible Kaggle competition data & the ICDAR 2023 competition data.



## Model Weights

> https://www.kaggle.com/datasets/navinkumarmnk/benetech-matcha-models
 
## Results
![Alt text](assets/results_matcha.png?raw=true "Results")
