import time
import random
def filecreator():
    open("finalprojectdata/digitdata/digitdatatrainer.txt", 'w').close()
    open("finalprojectdata/digitdata/digitdatalabeler.txt", 'w').close() 
    facedatatemp = open("finalprojectdata/digitdata/trainingimages","r")
    facedata = open("finalprojectdata/digitdata/digitdatatrainer.txt","a")
    facedata.write(facedatatemp.read())
    facedatatemp = open("finalprojectdata/digitdata/validationimages","r")
    facedata = open("finalprojectdata/digitdata/digitdatatrainer.txt","a")
    facedata.write(facedatatemp.read())

    facedatatemp = open("finalprojectdata/digitdata/traininglabels","r")
    facedata = open("finalprojectdata/digitdata/digitdatalabeler.txt","a")
    facedata.write(facedatatemp.read())
    facedatatemp = open("finalprojectdata/digitdata/validationlabels","r")
    facedata = open("finalprojectdata/digitdata/digitdatalabeler.txt","a")
    facedata.write(facedatatemp.read())
    open("finalprojectdata/digitdata/digitdatatest.txt", 'w').close()
    open("finalprojectdata/digitdata/digitlabeltestlabeler.txt", 'w').close() 
    
    facedatatemp = open("finalprojectdata/digitdata/testimages","r")
    facedata = open("finalprojectdata/digitdata/digitdatatest.txt","a")
    facedata.write(facedatatemp.read())

    facedatatemp = open("finalprojectdata/digitdata/testlabels","r")
    facedata = open("finalprojectdata/digitdata/digitlabeltestlabeler.txt","a")
    facedata.write(facedatatemp.read())
def trainer(height,width,cutoff,n):
    bad = 0
    total =0
    passthrough =0
    ws = [[((random.random())*((-1.0)**random.randint(1,2)))for _ in range(height*width)] for _ in range(10)]
    
    scores = [0.0 for _ in range(10)]
    #image =['' for i in range(70)]
    data = open("finalprojectdata/digitdata/digitdatatrainer.txt","r")
    label = open("finalprojectdata/digitdata/digitdatalabeler.txt","r")
    label = label.read().split("\n")
    w0 = [((random.random())*((-1)**random.randint(1,2)))for _ in range(10)]
    
    regions = [0.0 for _ in range(height*width)]
    data=data.read().split("\n")
    while total==0 or (bad*1.0/total>=cutoff and passthrough<=120):
        #print("run")
        passthrough+=1
        # if passthrough%15==0:
        #     print("run %s bad %s height %s width %s" %(passthrough,bad,height,width))
        bad =0
        total =0
        endface = 28
        linecounter =0
        labelindex = 0
        for dataline in data:
            linecounter+=1
            if(linecounter>endface):
                a =int(label[labelindex])
                # regions /= 4200
                # regions *= (5*width)
                regions[:] = [(x*height*width) /784 for x in regions]      
                num =0      
                ans = [-1, -5.0]  
                for wn in ws:
                    wscore=w0[num]
                    for w in range(len(regions)):
                        wscore+=wn[w]*regions[w]
                        # if(wscore>ans[1]):
                        #     ans = [num, wscore] 
                    scores[num]=wscore
                    if(wscore>=0 and a!=num):
                        for j in range(len(wn)):
                            wn[j] -= (regions[j])
                        w0[num]-=1
                        # print(wscore)
                    elif (wscore<0 and a==num):
                        for j in range(len(wn)):
                            wn[j] += (regions[j])
                        w0[num]+=1
                        # print(wscore)
                    num+=1
                #if(label[labelindex])!=ans[0]:
               
                b = scores.index(max(scores))
               # if passthrough==2:
                    
             #       print("%s %s" %(a,b))
                if(b!=a):
                    bad+=1    
                total+=1
                endface+=28
                labelindex+=1
                regions = [0.0 for _ in range(height*width)]
            #image[(linecounter-1)%70]= dataline
            l = ((linecounter-1)%28)*height//28
            #boolh=False
            for c in range(len(dataline)):
               # if(linecounter==74 and (c==0 or c== 59)):
                   # boolh = True
                    #print(l*width+c*width//60)
                if dataline[c]=='#' or dataline[c] == '+':
                    regions[l*width+c*width//28]+=1
                   # y= 5
                # if boolh:
                #     print(c)
                #     boolh=False
            if linecounter>(n/100.0)*168000:
                break
  #  print("hi")  
    return w0,ws,height,width
def tester(w0,ws,height,width):  

    data = open("finalprojectdata/digitdata/digitdatatest.txt","r")
    label = open("finalprojectdata/digitdata/digitlabeltestlabeler.txt","r")
    scores = [0.0 for _ in range(10)]
    label = label.read().split("\n")
    regions = [0.0 for _ in range(height*width)]
    data=data.read().split("\n")
    bad =0
    total =0
    endface = 28
    linecounter =0
    labelindex = 0
    for dataline in data:
        linecounter+=1
        if(linecounter>endface): 
            regions[:] = [(x*height*width) /700 for x in regions]      
            num =0      
           # ans = [-1, -5.0]       
            a = int(label[labelindex])     
            for wn in ws:
                wscore=w0[num]
                for w in range(len(regions)):
                    wscore+=wn[w]*regions[w]
                scores[num]=wscore 
                num+=1
            b = scores.index(max(scores))

            if a!=b:
                bad+=1   
            total+=1
            endface+=28
            labelindex+=1
            regions = [0 for _ in range(height*width)]
        l = ((linecounter-1)%28)*height//28
        for c in range(len(dataline)):
            if dataline[c]=='#' or dataline[c] == '+':
                regions[l*width+c*width//28]+=1
    #print((1-(bad*1.0/total))*100.0)
    return (1-(bad*1.0/total))*100.0

def main():
    filecreator()
    for i in range(10):
        tic = time.time()
        [w0,ws,height,width] = trainer(7,7,0.2,10*(i+1))
        accuracy = tester(w0,ws,height,width)
        toc = time.time()
        print("Percentage data: %s Accuracy: %s Execution time: %s"  %(10*(i+1), round(accuracy,2), round((toc-tic),2)))
        
main()




