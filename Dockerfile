FROM continuumio/miniconda3:latest

COPY . /app

WORKDIR /app

RUN conda install -y pandas numpy matplotlib scikit-learn python-graphviz seaborn pydotplus jupyter bokeh

ENTRYPOINT [ "bokeh", "serve", "entropy_slider.py", "--allow-websocket-origin=*" ]
