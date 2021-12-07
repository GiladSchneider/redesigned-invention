import time
import random
def filecreator():
    open("finalprojectdata/facedata/facedatatrainer.txt", 'w').close()
    open("finalprojectdata/facedata/facedatalabeler.txt", 'w').close() 
    facedatatemp = open("finalprojectdata/facedata/facedatatrain","r")
    facedata = open("finalprojectdata/facedata/facedatatrainer.txt","a")
    facedata.write(facedatatemp.read())
    facedatatemp = open("finalprojectdata/facedata/facedatavalidation","r")
    facedata = open("finalprojectdata/facedata/facedatatrainer.txt","a")
    facedata.write(facedatatemp.read())

    facedatatemp = open("finalprojectdata/facedata/facedatatrainlabels","r")
    facedata = open("finalprojectdata/facedata/facedatalabeler.txt","a")
    facedata.write(facedatatemp.read())
    facedatatemp = open("finalprojectdata/facedata/facedatavalidationlabels","r")
    facedata = open("finalprojectdata/facedata/facedatalabeler.txt","a")
    facedata.write(facedatatemp.read())

    open("finalprojectdata/facedata/facedatatest.txt", 'w').close() 
    facedatatemp = open("finalprojectdata/facedata/facedatatest","r")
    facedata = open("finalprojectdata/facedata/facedatatest.txt","a")
    facedata.write(facedatatemp.read())
    open("finalprojectdata/facedata/facelabeltest.txt", 'w').close() 
    facedatatemp = open("finalprojectdata/facedata/facedatatestlabels","r")
    facedata = open("finalprojectdata/facedata/facelabeltest.txt","a")
    facedata.write(facedatatemp.read())

def trainer(height,width,cutoff,num):
    filecreator()
    bad = 0
    total =0
    passthrough =0
    ws = [((random.random())*((-1.0)**random.randint(1,2))) for _ in range(height*width)]
    #image =['' for i in range(70)]
    data = open("finalprojectdata/facedata/facedatatrainer.txt","r")
    label = open("finalprojectdata/facedata/facedatalabeler.txt","r")
    label = label.read().split("\n")
    w0 = ((random.random())*((-1)**random.randint(1,2)))
    regions = [0.0 for _ in range(height*width)]
    data=data.read().split("\n")
    while total==0 or (bad*1.0/total>=cutoff and passthrough<=90):
        #print("run")
        passthrough+=1
        # if passthrough%15==0:
        #     print("run %s bad %s height %s width %s" %(passthrough,bad,height,width))
        bad =0
        total =0
        endface = 70
        linecounter =0
        labelindex = 0
        for dataline in data:
            linecounter+=1
            if(linecounter>endface):
                # regions /= 4200
                # regions *= (5*width)
                regions[:] = [(x*height*width) /4200 for x in regions]              
                wscore=w0
                for w in range(len(ws)):
                    wscore+=ws[w]*regions[w]
                if(wscore>=0 and int(label[labelindex])==0):
                    for j in range(len(ws)):
                        ws[j] -= (regions[j])
                    w0-=1
                    # print(wscore)

                    bad+=1
                elif (wscore<0 and int(label[labelindex])==1):
                    for j in range(len(ws)):
                        ws[j] += (regions[j])
                    w0+=1
                    # print(wscore)
                    bad+=1
                total+=1
                endface+=70
                labelindex+=1
                regions = [0.0 for _ in range(height*width)]
            #image[(linecounter-1)%70]= dataline
            l = ((linecounter-1)%70)*height//70
            #boolh=False
            for c in range(len(dataline)):
               # if(linecounter==74 and (c==0 or c== 59)):
                   # boolh = True
                    #print(l*width+c*width//60)
                if dataline[c]=='#' or dataline[c] == '+':
                    regions[l*width+c*width//60]+=1
                   # y= 5
                # if boolh:
                #     print(c)
                #     boolh=False
            if linecounter>(num/100.0)*52640:
                break
  #  print("hi")  
    return w0,ws,height,width
def tester(w0,ws,height,width):
    
    data = open("finalprojectdata/facedata/facedatatest.txt","r")
    label = open("finalprojectdata/facedata/facelabeltest.txt","r")
    stderror = []
    label = label.read().split("\n")
    regions = [0.0 for _ in range(height*width)]
    data=data.read().split("\n")
    bad =0
    total =0
    endface = 70
    linecounter =0
    labelindex = 0
    for dataline in data:
        linecounter+=1
        if(linecounter>endface):            
            wscore=w0
            for w in range(len(ws)):
                wscore+=ws[w]*regions[w]
            if(wscore>=0 and int(label[labelindex])==0) or (wscore<0 and int(label[labelindex])==1):
                bad+=1
            total+=1
            endface+=70
            labelindex+=1
            regions = [0 for _ in range(height*width)]
        l = ((linecounter-1)%70)*height//70
        for c in range(len(dataline)):
            if dataline[c]=='#' or dataline[c] == '+':
                regions[l*width+c*width//60]+=1
    #print((1-(bad*1.0/total))*100.0)
    return (1-(bad*1.0/total))*100.0

def main():
    for i in range(10):
        tic = time.time()
        [w0,ws,height,width] = trainer(7,7,0.19,(10*(1+i)))
        toc = time.time()
        accuracy = tester(w0,ws,height,width)
        print("Percentage data: %s Accuracy: %s Execution time: %s"  %(10*(i+1), round(accuracy,2), round((toc-tic),2)))
main()




