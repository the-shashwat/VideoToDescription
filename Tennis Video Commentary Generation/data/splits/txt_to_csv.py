text_file = "POINTS.txt"
csv_file = "POINTS.csv"
cfd= open(csv_file,"w")
cfd.write("pid,set,vid,start,end,matchpoint")
cfd.write("\n")
with open(text_file,"r") as fd:
    content = fd.readlines()
    for line in content:
        pid,info = line.split("\t")
        cfd.write(pid+","+info) 
cfd.close()
