# Drawing plots
rm(list = ls())
size = c("50,000", "100,000", "500,000", "1,000,000", "1,300,000")

# nbsvm mse and runtime
mse_nbsvm = c(0.7150445680609622,
        0.6426286404267321,
        0.5224403414518084,
        0.4904429653903459,
        0.4805430293789442)

runtime_nbsvm = c(164.34093689918518,
         229.50440788269043,
         701.6827099323273,
         1274.3079099655151,
         1810.5344378948212)

mse_nbsvm = rbind(mse_nbsvm,
            c(0.7150467476547174,
              0.6426303992865129,
              0.5224398920657443,
              0.49044258786664274,
              0.48054297755102626)
            )

runtime_nbsvm = rbind(runtime_nbsvm,
                c(159.19883584976196,
                  216.22107005119324,
                  691.0111610889435,
                  1281.946286201477,
                  1644.5932247638702)
                )

mse_nbsvm_clean = c(0.7540073401451344,
              0.7019338743208169,
              0.6050078373744816,
              0.5703639850663288,
              0.5594166814458159)

runtime_nbsvm_clean = c(62.340609073638916,
                  95.78064584732056,
                  416.80766677856445,
                  783.3386092185974,
                  1065.070950269699)

max_memory_nbsvm_clean = c(3556.1328125,
                           4016.16796875,
                           4883.0625,
                           6889.4453125,
                           8415.2421875)

# nb mse and runtime
mse_nb_clean = c(1.17307,
           1.11541,
           1.0309766666666667,
           1.02883,
           1.0323966666666666)

runtime_nb_clean = c(39.61942267417908,
               50.65673112869263,
               125.707590341568,
               219.24542045593262,
               269.718519449234)

max_memory_nb_clean = c(3346.27734375,
                        3552.890625,
                        5202.78515625,
                        7011.89453125,
                        8056.6328125)

mse_nb = c(1.0439866666666666,
           0.9953933333333334,
           0.9796866666666667,
           1.00215,
           1.0139866666666666)

runtime_nb = c(83.33110427856445,
               99.85633492469788,
               240.28191924095154,
               379.5233881473541,
               479.7275142669678)

max_memory_nb = c(3934.06640625,
                  4130.7109375,
                  5563.7578125,
                  8241.3125,
                  9699.23046875)


# LSTM mse and runtime
mse_lstm = c(1.40391,
             0.971385,
             0.624521,
             0.507794,
             0.358527)

runtime_lstm = c(831.241733,
                 1386.662537,
                 5965.158794,
                 11154.41401,
                 14388.01753)

max_memory_lstm = c(4279.773438,
                    3417.707031,
                    7244.527344,
                    9501.976563,
                    11828.60938)

# logistic regression data
logit = read.csv("./Logit_result/logit_result_clean.csv")


# comparison between cleaned and uncleaned data
# it is interesting that the cleaned data does worse
jpeg("./Plots/NBSVM_unclean_clean.jpeg")
plot(1:5,apply(mse_nbsvm,2,mean), type = 'b', main=NULL, xlab="Trainsize", ylab="MSE",
     ylim = c(0.4,0.9), xaxt="n", lwd=2,cex.lab=1.2)
axis(1, at=1:5, labels=size)
lines(1:5,mse_nbsvm_clean, type = 'b', col="red", lty=2, lwd=2)
legend("topright",legend=c("cleaned","uncleaned"), col=c("red","black"),lty=c(2,1), lwd=2)
dev.off()

# plot(1:5,mse_nb, type = 'b', main="MSE on of NB different train size", xlab="Trainsize", ylab="MSE",
#      ylim = c(0.4,1.2), xaxt="n")
# axis(1, at=1:5, labels=size)
# lines(1:5,mse_nb_clean, type = 'b', col="red", lty=2)
# legend("bottomright",legend=c("cleaned","uncleaned"), col=c("red","black"),lty=c(2,1))

# runtime
jpeg("./Plots/NBSVM_runtime_sample.jpeg")
plot(1:5,apply(runtime_nbsvm,2,mean), type = 'b', main=NULL, xlab="Trainsize", 
     ylim=c(50,2000), ylab="Time", xaxt="n", lwd=2,cex.lab=1.2)
axis(1, at=1:5, labels=size)
lines(1:5,runtime_nbsvm_clean, type = 'b', col="red", lty=2, lwd=2)
legend("topleft",legend=c("cleaned","uncleaned"), col=c("red","black"),lty=c(2,1), lwd=2)
dev.off()

# plot(1:5,runtime_nb, type = 'b', main="Runtime on of NB different train size", xlab="Trainsize", 
#      ylim=c(50,2000), ylab="Time", xaxt="n")
# axis(1, at=1:5, labels=size)
# lines(1:5,runtime_nb_clean, type = 'b', col="red", lty=2)
# legend("topleft",legend=c("cleaned","uncleaned"), col=c("red","black"),lty=c(2,1))

# scaled memory usage
memory_nbsvm1=(read.table("./NBSVM_memory/nbsvm0.txt",sep=","))
memory_nbsvm2=read.table("./NBSVM_memory/nbsvm_new0.txt",sep=",")
memory_nbsvm_clean=read.table("./NBSVM_memory/nbsvm_new_clean0.txt",sep=",")
memory_nb_clean=read.table("./NB_memory/nb0.txt",sep=",")

plot(seq(0,1,length.out = dim(memory_nbsvm1)[1]),t(memory_nbsvm1), type = 'l', ylim = c(1500,5000),
     main="A strange example on two runs of memory\n with sample size of 50000",
     ylab="memory",xlab = 'Scaled time')
lines(seq(0,1,length.out = dim(memory_nbsvm2)[1]),t(memory_nbsvm2))
lines(seq(0,1,length.out = dim(memory_nbsvm3)[1]),t(memory_nbsvm3))
lines(seq(0,1,length.out = dim(memory_nb)[1]),t(memory_nb))

# comparison between different methods

# mse
jpeg("./Plots/Compare_mse.jpeg")
plot(1:5,mse_nbsvm_clean, type = 'b', main=NULL, xlab="Trainsize", ylab="MSE",
     ylim = c(0.3,1.5), xaxt="n", lwd=2,cex.lab=1.2)
axis(1, at=1:5, labels=size)
lines(1:5,mse_nb_clean, type = 'b', col="red", lty=2, lwd=2)
lines(1:5,mse_lstm, type = 'b', col="deepskyblue2", lty = 3, lwd=2)
lines(1:5,logit$MSE, type = 'b', col="darkblue", lty = 4, lwd=2)
legend("topright",legend=c("LR","LSTM","NB","NBSVM"),
       col=c("darkblue","deepskyblue2","red","black"),lty=c(4,3,2,1), lwd=2)
dev.off()

table = rbind(logit$MSE,mse_nb_clean,mse_nbsvm_clean,mse_lstm)
rownames(table) = c("LR","NB","NBSVM","LSTM")
colnames(table) = size

write.csv(format(table,digits = 4),file = "./mse_table.txt")

# time
jpeg("./Plots/Compare_runtime.jpeg")
plot(1:5,runtime_nbsvm_clean, type = 'b', main=NULL, xlab="Trainsize", 
     ylim=c(50,15000), ylab="Time(s)", xaxt="n", lwd=2,cex.lab=1.2)
axis(1, at=1:5, labels=size)
lines(1:5,runtime_nb_clean, type = 'b', col="red", lty=2, lwd=2)
lines(1:5,runtime_lstm, type = 'b', col="deepskyblue2", lty = 3, lwd=2)
lines(1:5,logit$Time.sec., type = 'b', col="darkblue", lty = 4, lwd=2)
legend("topleft",legend=c("LR","NB","NBSVM","LSTM"),
       col=c("darkblue","red","black","deepskyblue2"),lty=c(4,2,1,3), lwd=2)
dev.off()

table = rbind(logit$Time.sec.,runtime_nb_clean,runtime_nbsvm_clean,runtime_lstm)
rownames(table) = c("LR","NB","NBSVM","LSTM")
colnames(table) = size

write.csv(format(table,digits = 4),file = "./runtime_table.txt")

# memory
jpeg("./Plots/Compare_max_memory.jpeg")
plot(1:5,max_memory_nbsvm_clean, type = 'b', main=NULL, xlab="Trainsize", 
      ylab="Memory(MB)", xaxt="n",ylim = c(1000,12000), lwd=2,cex.lab=1.2)
axis(1, at=1:5, labels=size)
lines(1:5,max_memory_nb_clean, type = 'b', col="red", lty=2, lwd=2)
lines(1:5,max_memory_lstm, type = 'b', col="deepskyblue2", lty = 3, lwd=2)
lines(1:5,logit$Maximum.memory.usage, type = 'b', col="darkblue", lty = 4, lwd=2)
legend("topleft",legend=c("LR","NB","NBSVM","LSTM"),
       col=c("darkblue","red","black","deepskyblue2"),lty=c(4,2,1,3), lwd=2)
dev.off()


