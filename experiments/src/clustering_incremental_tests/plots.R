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
  df$Conditions[df$Conditions == "Gewoon"] <- "minLB"
  df$Problem <- toupper(df$Problem)
  colnames(df)
  View(df)
  
  gap <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = RealGap,group=Conditions,color=Conditions)) + 
    geom_point(aes(y=RealGap,shape=Conditions,color=Conditions)) +
    facet_wrap(.~Problem, scales = "free_y", nrow=1) +
    scale_shape_manual(values = c(16,17,18,19)) +
    scale_linetype_manual(values = c("dashed","solid","dashed","solid")) +
    scale_color_manual(values=c("darkgreen", "darkblue", "#56B4E9")) +
    ylab("Gap") +
    xlab("Width")  +
    ggtitle("")
  
  error <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = MergeQuality, group=Conditions,color=Conditions)) + 
    geom_point(aes(y=MergeQuality,shape=Conditions,color=Conditions)) +
    facet_wrap(.~Problem, scales = "free_y", nrow=1) +
    scale_shape_manual(values = c(16,17,18,19)) +
    scale_linetype_manual(values = c("dashed","solid","dashed","solid")) +
    scale_color_manual(values=c("darkgreen", "darkblue", "#56B4E9")) +
    ylab("Merge Error") +
    xlab("Width")  +
    ggtitle("")
  
  finalplot <- ggarrange(gap + 
              theme(panel.border = element_rect(color = "black",
                                                fill = NA,
                                                size = 0.1),
                    axis.title = element_text(size = 18,face="bold"),
                    axis.text.x = element_text(size = 12,color="black"),
                    axis.text.y = element_text(size = 12,color="black"),
                    plot.title = element_text(size = 18,hjust = 0.5),
                    strip.text.x = element_text(size = 16, color = "black", face = "bold"),
                    strip.text.y = element_text(size = 16, color = "black", face = "bold"),
                    legend.position="right",
                    legend.title=element_blank(),
                    legend.text=element_text(size=12)),
            error + 
              theme(panel.border = element_rect(color = "black",
                                                fill = NA,
                                                size = 0.1),
                    axis.title = element_text(size = 18,face="bold"),
                    axis.text.x = element_text(size = 12,color="black"),
                    axis.text.y = element_text(size = 12,color="black"),
                    plot.title = element_text(size = 18,hjust = 0.5),
                    strip.text.x = element_text(size = 16, color = "black", face = "bold"),
                    strip.text.y = element_text(size = 16, color = "black", face = "bold"),
                    legend.position="right",
                    legend.title=element_blank(),
                    legend.text=element_text(size=12)), 
            nrow = 2)
  finalplot
  ggsave(plot = finalplot, file = "~/Documents/PhD/Diagrams/ComparativeDD_Paper/cluster.pdf")
}
plot_cluster()



plot_cluster_raw <- function(){
  # load data
  df <-read.csv("~/Documents/PhD/ddo/experiments/all_['Cluster', 'Gewoon'].csv")
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

  df$Conditions[df$Conditions == "Gewoon"] <- "minLB"
  df$Conditions[df$Conditions == "Dominance"] <- "Dominance+minLB"
  df$Problem <- toupper(df$Problem)
  View(df)
  # df <- subset(df, Solver == "incremental")
  df <- subset(df, Solver == "top-down")
  gap <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = RealGap,color=Conditions,linetype=Conditions)) + 
    geom_point(aes(y=RealGap,shape=Conditions,color=Conditions)) +
    facet_wrap(Problem~., scales = "free_y", nrow=1) +
    scale_shape_manual(values = c(16,17,18,19)) +
    scale_linetype_manual(values = c("dashed","solid","dashed","solid")) +
    scale_color_manual(values=c("darkgreen", "darkblue", "red","purple")) +
    ylab("Gap") +
    xlab("Width")  +
    ggtitle("") + 
    theme(panel.border = element_rect(color = "black",
                                      fill = NA,
                                      size = 0.1),
          axis.title = element_text(size = 18,face="bold"),
          axis.text.x = element_text(size = 12,color="black"),
          axis.text.y = element_text(size = 12,color="black"),
          plot.title = element_text(size = 18,hjust = 0.5),
          strip.text.x = element_text(size = 16, color = "black", face = "bold"),
          strip.text.y = element_text(size = 16, color = "black", face = "bold"),
          legend.position="right",
          legend.title=element_blank(),
          legend.text=element_text(size=12))
  
  gap
  ggsave(plot = gap, file = "~/Documents/PhD/Diagrams/ComparativeDD_Paper/varordcluster.pdf")
}

plot_cluster_dominance <- function(){
  # load data
  df <-read.csv("~/Documents/PhD/ddo/experiments/all_['Dominance', 'Dominance+Cluster', 'Gewoon', 'Cluster'].csv")
  colnames(df)
  df$Conditions[df$Conditions == "Gewoon"] <- "minLB"
  df$Conditions[df$Conditions == "Dominance"] <- "Dominance+minLB"
  df$Problem <- toupper(df$Problem)
  View(df)
  df <- subset(df, Solver == "incremental")
  # df <- subset(df, Solver == "top-down")
  gap <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = RealGap,color=Conditions,linetype=Conditions)) + 
    geom_point(aes(y=RealGap,shape=Conditions,color=Conditions)) +
    facet_wrap(Problem~., scales = "free_y", nrow=1) +
    scale_shape_manual(values = c(16,17,18,19)) +
    scale_linetype_manual(values = c("dashed","solid","dashed","solid")) +
    scale_color_manual(values=c("darkgreen", "darkblue", "red","purple")) +
    ylab("Gap") +
    xlab("Width")  +
    ggtitle("") + 
    theme(panel.border = element_rect(color = "black",
                                      fill = NA,
                                      size = 0.1),
          axis.title = element_text(size = 18,face="bold"),
          axis.text.x = element_text(size = 12,color="black"),
          axis.text.y = element_text(size = 12,color="black"),
          plot.title = element_text(size = 18,hjust = 0.5),
          strip.text.x = element_text(size = 16, color = "black", face = "bold"),
          strip.text.y = element_text(size = 16, color = "black", face = "bold"),
          legend.position="right",
          legend.title=element_blank(),
          legend.text=element_text(size=12))
  
  gap
  ggsave(plot = gap, file = "~/Documents/PhD/Diagrams/ComparativeDD_Paper/clusterdomIR.pdf")
}
plot_cluster_dominance()

plot_varOrd_dominance <- function(){
  # load data
  df <-read.csv("~/Documents/PhD/ddo/experiments/all_['Dominance', 'VarOrd', 'Dominance+VarOrd', 'Gewoon'].csv")
  df$Conditions[df$Conditions == "Gewoon"] <- "minLB"
  df$Conditions[df$Conditions == "Dominance"] <- "Dominance+minLB"
  df$Problem <- toupper(df$Problem)
  View(df)
  # df <- subset(df, Solver == "incremental")
  df <- subset(df, Solver == "top-down")
  gap <- ggplot(df, aes(x=Width)) +
    geom_line(aes(y = RealGap,color=Conditions,linetype=Conditions)) + 
    geom_point(aes(y=RealGap,shape=Conditions,color=Conditions)) +
    facet_wrap(Problem~., scales = "free_y", nrow=1) +
    scale_shape_manual(values = c(16,17,18,19)) +
    scale_linetype_manual(values = c("dashed","solid","dashed","solid")) +
    scale_color_manual(values=c("darkgreen", "darkblue", "red","purple")) +
    ylab("Gap") +
    xlab("Width")  +
    ggtitle("") + 
    theme(panel.border = element_rect(color = "black",
                                      fill = NA,
                                      size = 0.1),
          axis.title = element_text(size = 18,face="bold"),
          axis.text.x = element_text(size = 12,color="black"),
          axis.text.y = element_text(size = 12,color="black"),
          plot.title = element_text(size = 18,hjust = 0.5),
          strip.text.x = element_text(size = 16, color = "black", face = "bold"),
          strip.text.y = element_text(size = 16, color = "black", face = "bold"),
          legend.position="right",
          legend.title=element_blank(),
          legend.text=element_text(size=12))
  
  gap
  ggsave(plot = gap, file = "~/Documents/PhD/Diagrams/ComparativeDD_Paper/varorddom.pdf")
}

plot_cluster()
plot_cluster_dominance()
plot_dominance()
plot_varOrd()
plot_varOrd_cluster()
plot_varOrd_dominance()
