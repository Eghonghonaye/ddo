#Activate libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate) #for date
library(anytime)
library(reshape2)
library(Rcpp)
library(cowplot)
library(gtable)
library(grid)
library(gridExtra)
library(egg)
library(svglite)
# install.packages("svglite") 

theme = theme_set(theme_minimal())
theme = theme_update(legend.position="right", 
                     legend.title=element_text(),
                     panel.spacing = unit(2, "lines"))
par(mfrow=c(1,2))

plot_cluster <- function(){
  # load data
  df <-read.csv("~/Documents/PhD/ddo/experiments/all.csv")
  colnames(df)
  View(df)
  
  gap <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = RealGap,group=Conditions,color=Conditions)) + 
    facet_wrap(. ~ Problem, scales = "free_y", nrow=1) +
    scale_color_manual(values=c("darkgreen", "darkblue", "#56B4E9")) +
    ylab("Optimality Gap") +
    xlab("")  +
    ggtitle("")
  
  error <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = MergeQuality, group=Conditions,color=Conditions)) + 
    facet_wrap(. ~ Problem, scales = "free_y", nrow=1) +
    scale_color_manual(values=c("darkgreen", "darkblue", "#56B4E9")) +
    ylab("Merge Error") +
    xlab("")  +
    ggtitle("")
  
  
  ggarrange(gap + 
              theme(panel.border = element_rect(color = "grey",
                                                fill = NA,
                                                size = 0.1),
                    axis.title = element_text(size = 16),
                    axis.text.x = element_text(size = 8),
                    axis.text.y = element_text(size = 8),
                    plot.title = element_text(size = 16,hjust = 0.5),
                    legend.position="top",
                    legend.title=element_blank()), 
            error + 
              theme(axis.title = element_text(size = 16),
                    axis.text.x = element_text(size =8),
                    axis.text.y =  element_text(size = 8),
                    axis.ticks.y = element_blank(),
                    plot.margin = margin(r=1,l=1),
                    legend.position="none",
                    legend.title=element_blank(),
                    panel.border = element_rect(color = "grey",
                                                fill = NA,
                                                size = 0.1),
                    plot.title = element_text(size = 16,hjust = 0.5)
                    ), 
            nrow = 2)
}

plot_dominance <- function(){
  # load data
  df <-read.csv("~/Documents/PhD/ddo/experiments/all_['Dominance', 'Gewoon'].csv")
  colnames(df)
  View(df)
  
  gap <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = RealGap,color=Label)) + 
    facet_wrap(Problem~., scales = "free_y") +
    scale_color_manual(values=c("darkgreen", "darkblue", "red","purple")) +
    ylab("") +
    xlab("")  +
    ggtitle("Optimality Gap")
  
  gap
}

plot_varOrd <- function(){
  # load data
  df <-read.csv("~/Documents/PhD/ddo/experiments/all_['VarOrd', 'Gewoon'].csv")
  colnames(df)
  View(df)
  
  gap <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = RealGap,color=Conditions)) + 
    facet_wrap(Problem~., scales = "free_y") +
    scale_color_manual(values=c("darkgreen", "darkblue", "red","purple")) +
    ylab("") +
    xlab("")  +
    ggtitle("Optimality Gap")
  
  gap
}


plot_varOrd_cluster <- function(){
  # load data
  df <-read.csv("~/Documents/PhD/ddo/experiments/all_['Cluster', 'VarOrd', 'Cluster+VarOrd', 'Gewoon'].csv")
  colnames(df)
  View(df)
  
  gap <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = RealGap,color=Conditions)) + 
    facet_wrap(Problem~., scales = "free_y") +
    scale_color_manual(values=c("darkgreen", "darkblue", "red","purple")) +
    ylab("") +
    xlab("")  +
    ggtitle("Optimality Gap")
  
  gap
}

plot_cluster_dominance <- function(){
  # load data
  df <-read.csv("~/Documents/PhD/ddo/experiments/all_['Dominance', 'Dominance+Cluster', 'Gewoon', 'Cluster'].csv")
  colnames(df)
  View(df)
  df <- subset(df, Solver == "incremental")
  gap <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = RealGap,color=Conditions)) + 
    facet_wrap(Problem~., scales = "free_y") +
    scale_color_manual(values=c("darkgreen", "darkblue", "red","purple")) +
    ylab("") +
    xlab("")  +
    ggtitle("Optimality Gap")
  
  gap
}

plot_varOrd_dominance <- function(){
  # load data
  df <-read.csv("~/Documents/PhD/ddo/experiments/all_['Dominance', 'VarOrd', 'Dominance+VarOrd', 'Gewoon'].csv")
  colnames(df)
  View(df)
  
  gap <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = RealGap,color=Conditions)) + 
    facet_wrap(Problem~., scales = "free_y") +
    scale_color_manual(values=c("darkgreen", "darkblue", "red","purple")) +
    ylab("") +
    xlab("")  +
    ggtitle("Optimality Gap")
  
  gap
}

plot_cluster()
plot_cluster_dominance()
plot_dominance()
plot_varOrd()
plot_varOrd_cluster()
plot_varOrd_dominance()
