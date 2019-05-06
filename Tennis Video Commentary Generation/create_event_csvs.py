import json
import os

def identify_events(dir_path):
    save_path = "./data/annotations/"
    serve_fd = open(os.path.join(save_path,"serve.csv"),"w")
    serve_fd.write("name,vid,start,end,result,player")
    serve_fd.write("\n")
    hit_fd = open(os.path.join(save_path,"hit.csv"),"w")
    hit_fd.write("name,vid,start,end,side,player,type")
    hit_fd.write("\n")
    for f in os.listdir(dir_path):
        fd = open(os.path.join(dir_path,f))
        print "File => "+f
        json_data = json.load(fd)
        serve_events = json_data["classes"]["Serve"]
        for event in serve_events:
            start = event["start"]
            end = event["end"]
            name = event["name"]
            custom = event["custom"]
            #if f[0:4] == "V007" and name == "0081":
            #    print event
            serve_fd.write(name+","+f[0:4]+","+str(start)+","+str(end)+","+custom["Result"]+","+custom["Player"])
            serve_fd.write("\n")
        hit_events = json_data["classes"]["Hit"]
        for event in hit_events:
            start = event["start"]
            end = event["end"]
            name = event["name"]
            custom = event["custom"]
            hit_fd.write(name+","+f[0:4]+","+str(start)+","+str(end)+","+custom["Side"]+","+custom["Player"]+","+custom["Type"])
            hit_fd.write("\n")
        fd.close()
    serve_fd.close()
    hit_fd.close()


      
    
def main():
    dir_path = "./data/annotations/generalised/"
    identify_events(dir_path)

if __name__=="__main__":
    main()
