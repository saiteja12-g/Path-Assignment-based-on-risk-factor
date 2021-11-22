from PIL import Image
im = Image.open("download.jpeg").convert("L")
im1 = Image.Image.getcolors(im)
temp = []
count = 0 
for i in range (len(im1)):
    
    temp.append(im1[i][0])
    if i in temp:
        count = count +1

print(count)        

d = dict((x,temp.count(x)) for x in set(temp))
print(d)