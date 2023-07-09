FROM continuumio/anaconda3:latest
COPY . /usr/app/
WORKDIR /usr/app/
RUN conda env create -f umbra.yml
RUN echo "source activate umbra" > ~/.bashrc
CMD ["/bin/bash"]
