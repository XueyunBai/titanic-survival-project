FROM r-base:4.4.3
WORKDIR /app
COPY install_packages.R .
RUN Rscript install_packages.R
COPY main.R .
CMD ["Rscript", "main.R", "/data"]