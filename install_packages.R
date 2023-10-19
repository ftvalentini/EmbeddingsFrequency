
urls = c(
    "https://cran.r-project.org/src/contrib/tidyverse_2.0.0.tar.gz",
    "https://cran.r-project.org/src/contrib/rmarkdown_2.21.tar.gz",
    "https://cran.r-project.org/src/contrib/scales_1.2.1.tar.gz",
    "https://cran.r-project.org/src/contrib/glue_1.6.2.tar.gz",
    "https://cran.r-project.org/src/contrib/latex2exp_0.9.6.tar.gz",
    "https://cran.r-project.org/src/contrib/cowplot_1.1.1.tar.gz",
    "https://cran.r-project.org/src/contrib/janitor_2.2.0.tar.gz",
    "https://cran.r-project.org/src/contrib/texreg_1.38.6.tar.gz",
    "https://cran.r-project.org/src/contrib/remotes_2.4.2.tar.gz"
)


for (url in urls) {
    install.packages(url, repos=NULL, type="source", dependencies=T)
}

remotes::install_github(
    "rpkgs/gg.layers", ref="c30679fda2f829608abba93a7063544cfdd6a68b")
