#!/usr/bin/env python

import re
import glob
import sys

if len(sys.argv) != 2:
    print 'usage: python makeFunctionIndex.py directory'
    sys.exit(1)

path = str(sys.argv[1])

def getClassListNew():
    text = open(path + "/../../../vigra_build/docsrc/tagfile.tag").read()
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
    text = open(path + "/../../../vigra_build/docsrc/tagfile.tag").read()
    # all functions from compound "namespace", "file" and "group"
    # -> do not include member functions from compound "class"
    functions = re.findall(r'<compound kind="namespace">(.*?)</compound>',text,flags=re.DOTALL)

    #do not need these!? dublicates!?
    #functions += re.findall(r'<compound kind="group">(.*?)</compound>',text,flags=re.DOTALL)
    #functions += re.findall(r'<compound kind="file">(.*?)</compound>',text,flags=re.DOTALL)

    print functions[5]
    print len(functions)

    funcs = []
 
    for f in functions:
        funcs += re.findall(r'<member kind="function">.*?<name>(.*?)</name>.*?<anchorfile>(.*?)</anchorfile>.*?<anchor>(.*?)</anchor>.*?</member>',f,flags=re.DOTALL)

    function_list = []
    for f in funcs:
        function_list.append((f[1] + '#' + f[2], f[0]))

    # add special documentation for argument object factories
    for k in ['srcImageRange', 'srcImage', 'destImageRange', 'destImage', 'maskImage']:
        function_list.append(('group__ImageIterators.html#ImageBasedArgumentObjectFactories', k))
    for k in ['srcMultiArrayRange', 'srcMultiArray', 'destMultiArrayRange', 'destMultiArray']:
        function_list.append(('group__ImageIterators.html#MultiArrayBasedArgumentObjectFactories', k))
    for k in ['srcIterRange', 'srcIter', 'destIterRange', 'destIter', 'maskIter']:
        function_list.append(('group__ImageIterators.html#IteratorBasedArgumentObjectFactories', k))

    function_list.sort(lambda a,b: cmp(a[1], b[1]))
    function_list = disambiguateOverloadedFunctions(function_list)
    return function_list

def addHeading(index, initial):    
    index = index + '<p><a name="index_' + initial + \
    '"><table class="function_index"><tr><th> ' + initial.upper() + \
    ' </th><td align="right" width="100%">VIGRA_NAVIGATOR_PLACEHOLDER</td></tr></table><p>\n'
    return index

def disambiguateOverloadedFunctions(functionList):
    for i in xrange(len(functionList)):
        overloaded = False
        functionName = functionList[i][1]
        if i > 0:
            lastFunctionName = functionList[i-1][1]
            if functionName == lastFunctionName:
                overloaded = True
        if i < len(functionList) - 1:
            nextFunctionName = functionList[i+1][1]
            if functionName == nextFunctionName:
                overloaded = True
        if overloaded:
            # disambiguate overloaded functions by their group or namespace
            link = functionList[i][0]
            group = re.sub(r'(group__|namespacevigra_1_1)([^\.]+)\.html.*', r'\2', link)
        else:
            group = ""
        functionList[i] = functionList[i] + (group,)
    
    return functionList


def generateFunctionIndex(functionList):
    index = ""
    initials = []
    for i in range(len(functionList)):
        functionName = functionList[i][1]
        link = functionList[i][0]
        initial = functionName[0]
        if i > 0:
            lastInitial = functionList[i-1][1][0]
            if initial != lastInitial:
                initials.append(initial)
                index = addHeading(index, initial)
        else:
            initials.append(initial)
            index = addHeading(index, initial)
            
        index = index + '<a href="'+ link + '">' + functionName + '</a>()'
        overloadDisambiguation = functionList[i][2]
        if overloadDisambiguation != "":
            index = index + ' [' + overloadDisambiguation + ']'
        index = index + '<br>\n'

    navigator = '['
    for i in range(len(initials)):
        initial = initials[i]
        if i != 0:
            navigator = navigator + '|'
        navigator = navigator + ' <a href="#index_' + initial + '">' + initial.upper() + '</a> '
    navigator = navigator + ']'
    index = re.sub('VIGRA_NAVIGATOR_PLACEHOLDER', navigator, index)

    # use file "/namespaces.html" as boiler plate for "/functionindex.html"
    text = open(path + "/namespaces.html").read()
    if text.find('</h1>') > -1: # up to doxygen 1.7.1
        header = text[:text.find('</h1>')+5]
    else: # for doxygen 1.7.4 to 1.7.6.1
        header = text[:re.search(r'<div class="title">[^<]*</div>\s*</div>\s*</div>(?:<!--header-->)?\n<div class="contents">',text).end()]
    footer = re.search(r'(?s)(<!-- footer.html -->.*)', text).group(1)

    text = re.sub(r'Namespace List', r'Function Index', header)
    text = text + '\n<p><hr>\n'
    text = text + index
    text = text + footer

    open(path + "/functionindex.html", 'w+').write(text)



classList = getClassListNew()
functionList = getFunctionListNew()
generateFunctionIndex(functionList)

# Export class and function list to c_api_replaces.txt for 
# crosslinking of vigranumpy documentation.
# Note that '::' are not allowed in reStructuedText link names, 
# so we have to use '.' instead.
replaces=open("../vigranumpy/docsrc/c_api_replaces.txt","w")
for i in range(len(functionList)):
    functionName = functionList[i][1]
    overloadDisambiguation = functionList[i][2]
    if i > 0 and functionName == functionList[i-1][1] and \
                   overloadDisambiguation == functionList[i-1][2]:
        continue
    if overloadDisambiguation != "":
        functionName = overloadDisambiguation +'.' + functionName
    link = functionList[i][0]
    replaces.write(functionName+":"+link+"\n")
for i in range(len(classList)):
    className = classList[i][1]
    namespace = classList[i][2]
    if (i > 0 and className == classList[i-1][1]) or \
       (i < len(classList)-1 and className == classList[i+1][1]):
        namespace = namespace.replace('::', '.')
        className = namespace +'.' + className
    link = classList[i][0]
    replaces.write(className+":"+link+"\n")
replaces.close()
