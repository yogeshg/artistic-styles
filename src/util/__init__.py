import numpy as np

def about(x, LINE=80, SINGLE_LINE=False):
    '''
    author: Yogesh Garg (https://github.com/yogeshg)
    '''
    s ='type:'+str(type(x))+' '
    try:
        s+='shape:'+str(x.shape)+' '
    except Exception as e:
        pass
    try:
        s+='dtype:'+str(x.dtype)+' '
    except Exception as e:
        pass
    try:
        s1 = str(x)
        if(SINGLE_LINE):
            s1 = ' '.join(s1.split('\n'))
            extra = (len(s)+len(s1)) - LINE
            if(extra > 0):
                s1 = s1[:-(extra+3)]+'...'
            s+=s1
        else:
            s+='\n'+s1
    except Exception as e:
        pass
    return s

a = np.eye(3)
print about(a)
print about(a, SINGLE_LINE=True)

