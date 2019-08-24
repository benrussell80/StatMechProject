# StatMechProject (Spring 2019)
A visual guide to how information gain is used in tree-based classifiers. I trained decision-tree and random-forest classifiers on the pulsar star data set available from [Kaggle](https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star). I created visualizations of information gain and classification metrics in Bokeh and Jupyter Notebook. 

## Usage
Download

    git clone https://github.com/benrussell80/StatMechProject

### EDA and Results in Jupyter Notebook
The jupyter notebook `StatMechProject.ipynb` contains exploratory data analysis I performed on the data set using seaborn, matplotlib, and pandas.

I then trained a decision-tree classifier and a random-forest classifier from scikit-learn on the data set. I used visualization tools to investigate the inner workings of the classifiers and elucidate metrics such as feature importance, accuracy, recall, precision, and f1-score.

#### Create Conda Env and Open Notebook
Set up conda environment

    cd StatMechProject
    conda create -f environment.yml -n statmechproject
    conda activate statmechproject

Then

    jupyter-notebook

Open `StatMechProject.ipynb` and `info_gain_and_shannon_entropy.ipynb` at `localhost:8888`.

### Information Gain Visualization in Bokeh App
This bokeh app displays the data from the binary stars data set color-coded by their class. Each feature is displayed in a different tab and a plot of the information gain versus threshold accompanies it. A slider allows you to see how the values of each kind of entropy change with threshold value.

#### Open with Conda Env
With the same environment run

    bokeh serve entropy_slider.py

and open `localhost:5006`.

#### With Docker
Build image and run daemonized container. Expose bokeh app on port 5006.

    cd StatMechProject
    docker build -t statmechproject:latest .
    docker run -d -p 5006:5006 statmechproject:latest

Open `localhost:5006`

(image size ~3GB)
