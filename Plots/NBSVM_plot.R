# Drawing plots

size = c("50,000", "100,000", "500,000", "1,000,000", "1,300,000")

mse = c(0.7150445680609622,
        0.6426286404267321,
        0.5224403414518084,
        0.4904429653903459,
        0.4805430293789442)

runtime = c(164.34093689918518,
         229.50440788269043,
         701.6827099323273,
         1274.3079099655151,
         1810.5344378948212)

mse = rbind(mse,
            c(0.7150467476547174,
              0.6426303992865129,
              0.5224398920657443,
              0.49044258786664274,
              0.48054297755102626)
            )

runtime = rbind(runtime,
                c(159.19883584976196,
                  216.22107005119324,
                  691.0111610889435,
                  1281.946286201477,
                  1644.5932247638702)
                )

mse_clean = c(0.7540073401451344,
              0.7019338743208169,
              0.6050078373744816,
              0.5703639850663288,
              0.5594166814458159)
runtime_clean = c(62.340609073638916,
                  95.78064584732056,
                  416.80766677856445,
                  783.3386092185974,
                  1065.070950269699)

# it is interesting that the cleaned data does worse
plot(1:5,apply(mse,2,mean), type = 'b', main="MSE on of NBSVM different train size", xlab="Trainsize", ylab="MSE",
     ylim = c(0.4,0.8), xaxt="n")
axis(1, at=1:5, labels=size)
lines(1:5,mse_clean, type = 'b', col="red", lty=2)
legend("topright",legend=c("cleaned","uncleaned"), col=c("red","black"),lty=c(2,1))

# runtime
plot(1:5,apply(runtime,2,mean), type = 'b', main="Runtime on of NBSVM different train size", xlab="Trainsize", 
     ylim=c(50,2000), ylab="Time", xaxt="n")
axis(1, at=1:5, labels=size)
lines(1:5,runtime_clean, type = 'b', col="red", lty=2)
legend("topleft",legend=c("cleaned","uncleaned"), col=c("red","black"),lty=c(2,1))

# scaled memory usage

memory1=(read.table("./NBSVM_memory/nbsvm0.txt",sep=","))
memory2=read.table("./NBSVM_memory/nbsvm_new0.txt",sep=",")
memory3=read.table("./NBSVM_memory/nbsvm_new_clean0.txt",sep=",")

plot(seq(0,1,length.out = dim(memory1)[1]),t(memory1), type = 'l', ylim = c(1500,5000),
     main="A strange example on two runs of memory\n with sample size of 50000",
     ylab="memory",xlab = 'Scaled time')
lines(seq(0,1,length.out = dim(memory2)[1]),t(memory2))
lines(seq(0,1,length.out = dim(memory3)[1]),t(memory3))
