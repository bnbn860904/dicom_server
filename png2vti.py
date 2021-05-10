from paraview import simple
import os
import argparse 

#path = "C:/Users/leowang/Desktop/45"
   
def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="path to the file to convert", dest="input")
    parser.add_argument('--output', help="path to the file to convert", dest="output")
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = process_command()
    path = args.input
    outputpath = 'C:/Users/leowang/Desktop/' + args.output + '.vti'

a = os.listdir(path)
a.sort(key=lambda a: int(a[:-4]))
print(a)
k = len(a)
for i in range(0, k):
    a[i] = path + '/' + a[i]

imageStack = simple.PNGSeriesReader(
    FileNames= a,
    DataSpacing=[1,1,1])

#simple.SaveData('deleteMe777.vti', proxy=imageStack, CompressorType='ZLib')
simple.SaveData(outputpath, proxy=imageStack, CompressorType='ZLib')