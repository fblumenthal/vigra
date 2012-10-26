#!/usr/bin/env python

import re
import glob
import sys

if len(sys.argv) != 2:
    print 'usage: python makeFunctionIndex.py directory'
    sys.exit(1)

path = str(sys.argv[1])

def getClassListNew():
    text = open(path + "/tagfile_new.tag").read()
    # all classes except classes of global namespace
    classes = re.findall(r'<compound kind="class">\n.*?<name>(.*?::.*?)</name>\n.*?<filename>(.*?)</filename>',text)

    classes_plus_namespace = [[None, None, None] for k in xrange(len(classes))]

    for i in xrange(len(classes)):
        classes_plus_namespace[i][0]=classes[i][1]        
        classes_plus_namespace[i][1]=classes[i][0][classes[i][0].rfind('::')+2:]
        classes_plus_namespace[i][2]=classes[i][0][:classes[i][0].rfind('::')]
    
    classes_plus_namespace.sort(lambda a,b: cmp(a[1], b[1]))
    return classes_plus_namespace

def getFunctionListNew():
    text = open(path + "/tagfile_new.tag").read()
    # all functions from compound "namespace", "file" and "group"
    # -> do not include member functions from compound "class"
    functions = re.findall(r'<compound kind="group">(.*?)</compound>',text,flags=re.DOTALL)
    functions += re.findall(r'<compound kind="namespace">(.*?)</compound>',text,flags=re.DOTALL)
    functions += re.findall(r'<compound kind="file">(.*?)</compound>',text,flags=re.DOTALL)

    print functions[51]
    print len(functions)

    funcs = []
    
    func = re.findall(r'<member kind="function">.*?<name>(.*?)</name>.*?<anchorfile>(.*?)</anchorfile>.*?<anchor>(.*?)</anchor>.*?</member>',functions[50],flags=re.DOTALL)
    
    func += re.findall(r'<member kind="function">.*?<name>(.*?)</name>.*?<anchorfile>(.*?)</anchorfile>.*?<anchor>(.*?)</anchor>.*?</member>',functions[51],flags=re.DOTALL)

    print func

#    for f in functions:
#        funcs += re.findall(r'.*?<member kind="function">.*?<name>(.*?)</name>.*?<anchorfile>(.*?)</anchorfile>.*?<anchor>(.*?)</anchor>.*?<\member>',f,flags=re.DOTALL)
    
class_list = getClassListNew()

for c in class_list:
    print c[1]

print class_list[-1]
print class_list[0]
print class_list[30]
    
getFunctionListNew()
