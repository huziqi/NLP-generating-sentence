# -*- coding: utf-8 -*-
import numpy as np
import re

def delblankline(infile, outfile):
    infp=open(infile,"r")
    outfp=open(outfile,"w")
    #pattern=re.compile(r'[qwertyuiopasdfghjklzxcvbnm“”""|WQCＷＳＯ·.,:：《》!！？?()（）、；——0123456789*‘’]')
    pattern=re.compile(r'[qwertyuiopasdfghjklzxcvbnm]')
    lines=infp.readlines()
    for line in lines:
        if line.split():
            line=re.sub("\……", "",line)
            line=re.sub("\-+","",line)
            line=re.sub(pattern,"\0",line)
            line=re.sub("\ ","",line)
            if line!="\n":
                outfp.writelines(line)
    infp.close()
    outfp.close()

if __name__ == "__main__":
    delblankline("/home/guohf/cut.txt", "/home/guohf/cut2.txt")
    # delblankline("/home/guohf/AI_tutorial/ch8/data/old_man_and_sea.txt", "/home/guohf/AI_tutorial/ch8/data/old_man_and_sea2.txt")