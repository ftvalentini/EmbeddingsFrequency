
library(tidyverse)
library(latex2exp)

setwd("~/bias-frequency/")

# read csvs
files = c(
  "results/bias_resampling_gender.csv", 
  "results/bias_resampling_affluence.csv",
  "results/bias_resampling_ethnicity.csv"
)
dfs = list()
for (i in seq_along(files)) {
  dfs[[i]] = read_csv(files[i])
}

add_frequency_bins = function(df) {
  log_freq = log10(df[["freq"]])
  max_value = max(log_freq)
  cuts = unique(c(2, seq(2, 6., 1), max_value))
  df = df %>% mutate(freq_bin = cut(log_freq, cuts, include.lowest=T))
  return(df)
}

df = bind_rows(dfs) %>% 
  mutate(
    WE = factor(WE, levels=c("SGNS", "FastText", "GloVe")),
    type = factor(type, levels=c("Gender bias", "Ethnicity bias", "Affluence bias")),
  ) %>% 
  add_frequency_bins()

p = ggplot(df, aes(x=f10_b, y=bias, color=freq_bin)) +
  geom_hline(yintercept=0, linetype="dashed", linewidth=0.2, color="black") +
  stat_summary(fun.data="mean_cl_normal", geom="errorbar", width=.05) +
  stat_summary(fun="mean", geom="point") + 
  stat_summary(fun="mean", geom="line") + 
  facet_grid(vars(WE), vars(type), scales="free") +
  scale_color_viridis_d() +
  theme_light() +
  labs(
    x=TeX("$\\log_{10}\\,frequency_{B}$"),
    y=TeX("$Bias_{WE}(x,A,B)$"),
    color=TeX("$\\log_{10}\\,frequency_{x}$")
  ) +
  theme(
    legend.position="bottom", 
    strip.background=element_blank(), strip.placement="outside",
    legend.margin=margin(-5, -5, -5, -5),
    strip.text=element_text(colour="black", size=10)
  ) +
  NULL

ggsave(filename="results/plots/grid_impact.png", plot=p, height=5, width=8, dpi=300)
ggsave(filename="/home/fvalentini/wefrequency_arr_2023_april/img/grid_impact.png", plot=p, height=5, width=8, dpi=300)
