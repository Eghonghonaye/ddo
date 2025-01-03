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

# load data
df <-read.csv("~/Documents/PhD/ddo/experiments/all.csv")
colnames(df)
View(df)

gap <- ggplot(df, aes(x=Width)) +
  geom_line(aes(y = RealGap,group=Conditions,color=Conditions)) + 
  facet_grid(Problem~., scales = "free_y") +
  scale_color_manual(values=c("darkgreen", "darkblue", "#56B4E9")) +
  ylab("") +
  xlab("")  +
  ggtitle("Optimality Gap")

error <- ggplot(df, aes(x=Width)) +
  geom_line(aes(y = MergeQuality, group=Conditions,color=Conditions)) + 
  facet_grid(Problem~., scales = "free_y") +
  scale_color_manual(values=c("darkgreen", "darkblue", "#56B4E9")) +
  ylab("") +
  xlab("")  +
  ggtitle("Merge Error")



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
                  axis.title.y = element_blank(),
                  plot.margin = margin(r=1,l=1),
                  legend.position="top",
                  legend.title=element_blank(),
                  panel.border = element_rect(color = "grey",
                                              fill = NA,
                                              size = 0.1),
                  plot.title = element_text(size = 16,hjust = 0.5)), 
          nrow = 1)